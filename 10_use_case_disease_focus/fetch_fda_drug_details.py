import os
import requests
import pandas as pd
from tqdm import tqdm
import unicodedata
import re
import argparse
import time

def load_fda_api_key(path="fda_api_key.txt"):
    """Read the OpenFDA API key from a text file (key after '=')"""
    if not os.path.exists(path):
        print(f"[WARN] API key file '{path}' not found — continuing without key")
        return None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("fda_api_key"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key = parts[1].strip()
                    if key:
                        print(f"[INFO] Found API key in {path}")
                        return key
    print(f"[WARN] No valid 'fda_api_key=' entry found in {path}")
    return None

API_KEY = load_fda_api_key("fda_api_key.txt")

def first_fda_approval_details(drug):
    q = f'openfda.generic_name:"{drug}"'
    try:
        resp = requests.get(
            "https://api.fda.gov/drug/drugsfda.json",
            params={"search": q, "limit": 100, "api_key": API_KEY}
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 404:
            return None
        else:
            raise

    records = resp.json().get("results", [])

    orig_subs = []
    for rec in records:
        for sub in rec.get("submissions", []):
            if sub.get("submission_type") in ("ORIG", "NDA", "BLA"):
                date = pd.to_datetime(sub["submission_status_date"],
                                      format="%Y%m%d",
                                      errors="coerce")
                if pd.notna(date):
                    orig_subs.append({
                        "date": date,
                        "submission": sub,
                        "record": rec
                    })

    if not orig_subs:
        return None

    first = min(orig_subs, key=lambda x: x["date"])
    sub = first["submission"]
    rec = first["record"]
    openfda = rec.get("openfda", {})

    return {
        "generic_name": drug,
        "sponsor_name": rec.get("sponsor_name"),
        "application_number": rec.get("application_number"),
        "submission_type": sub.get("submission_type"),
        "submission_number": sub.get("submission_number"),
        "approval_date": first["date"],
        "indication": rec.get("products", [{}])[0].get("indication"),
        "pharm_class_cs": openfda.get("pharm_class_cs"),
        "pharm_class_epc": openfda.get("pharm_class_epc"),
        "pharm_class_pe": openfda.get("pharm_class_pe"),
        "pharm_class_moa": openfda.get("pharm_class_moa")
    }

def _normalize(s) -> str:
    if s is None:
        return ""
    try:
        s = str(s)
    except Exception:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    # allow letters/digits/hyphens; everything else → space
    s = re.sub(r'[^0-9a-z\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_approved_index(approved_list):
    # keep original + normalized for faster matching
    return [(orig, _normalize(orig)) for orig in approved_list]
    
def parse_args():
    ap = argparse.ArgumentParser(description="Fetch FDA approval metadata with checkpointing.")
    ap.add_argument("--output", default="out/fda_drug_metadata_progress.csv",
                    help="Path to progress/final CSV (checkpoints overwrite this).")
    ap.add_argument("--terms", default="out/unique_drug_terms_218510.csv",
                    help="CSV containing drug term candidates.")
    ap.add_argument("--min-articles", type=int, default=2,
                    help="Only consider terms that appear in at least this many articles.")
    ap.add_argument("--checkpoint-every", type=int, default=1000,
                    help="Save a checkpoint every N newly processed records.")
    ap.add_argument("--terms-col", default="drug_term_umls_norm_manual_clean",
                    help="Column name in --terms CSV containing the normalized drug term.")
    ap.add_argument("--articles-col", default="n_articles",
                    help="Column name in --terms CSV containing article counts.")
    return ap.parse_args()


# -----------------------------
# Checkpoint I/O
# -----------------------------
def resume_from_checkpoint(output_path):
    """
    Return (df_existing, processed_set). If output exists, load it and collect the
    set of canonical_drug_name values to skip. Else, return empty DataFrame & set.
    """
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        processed = set(df_existing.get("canonical_drug_name", pd.Series(dtype=str)).dropna().unique())
        print(f"[INFO] Resuming from checkpoint — {len(processed)} drugs already processed.")
    else:
        df_existing = pd.DataFrame()
        processed = set()
    return df_existing, processed


def save_csv_atomic(df, path):
    """
    Atomic write to prevent corruption if interrupted mid-save.
    """
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# -----------------------------
# Terms → Index
# -----------------------------
def load_terms_and_build_index(terms_path, articles_col, terms_col, min_articles):
    """
    Load the terms file, filter by min_articles, and build the approved index.
    """
    terms = pd.read_csv(terms_path)
    if articles_col not in terms.columns or terms_col not in terms.columns:
        raise KeyError(f"Columns not found in {terms_path}: "
                       f"missing {articles_col!r} or {terms_col!r}")

    # Filter for sufficiently common terms (prevents noisy one-offs)
    terms_common = terms[terms[articles_col] >= min_articles]
    terms_common_list = list(terms_common[terms_col])

    # Domain-specific: user-provided function that returns an iterable of (canonical_name, query_name)
    approved_index = build_approved_index(terms_common_list)
    return approved_index


LABEL_BASE = "https://api.fda.gov/drug/label.json"

def fetch_application_label_details(app_nr):
    params = {
        "search": f'openfda.application_number:"{app_nr}"',
        "limit": 1,
        "sort": "effective_time:desc",
        "api_key": API_KEY
    }
    
    res = requests.get(LABEL_BASE, params=params, timeout=30)
    data = res.json()
    
    if "results" not in data:
        #print("No results found or bad request.")
        return "No results found or bad request.", "No results found or bad request.", "No results found or bad request."
    else:
        result = data["results"][0]
    
        # --- Extract clinical studies text and find all NCT IDs
        clinical_text = " ".join(result.get("clinical_studies", []))
        nct_ids = re.findall(r"NCT\d{8}", clinical_text)
        
        # --- Extract only the first sentence from indications_and_usage
        indications_text = " ".join(result.get("indications_and_usage", []))
        match = re.match(r"^(.*?\.)\s", indications_text)
        first_sentence = match.group(1).strip() if match else indications_text.strip()

        brand_name = " ".join(result.get("brand_name", []))
         
        nct_ids_str = ", ".join(nct_ids) if nct_ids else "No NCTIDs found."

        return nct_ids_str, first_sentence, indications_text
        
def all_fda_approval_details(drug):
    q = f'openfda.generic_name:"{drug}"'
    try:
        resp = requests.get(
            "https://api.fda.gov/drug/drugsfda.json",
            params={"search": q, "limit": 100, "api_key": API_KEY}
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 404:
            return None
        else:
            raise

    records = resp.json().get("results", [])

    orig_subs = []
    for rec in records:
        for sub in rec.get("submissions", []):
            if sub.get("submission_type") in ("ORIG", "NDA", "BLA"):
                date = pd.to_datetime(sub["submission_status_date"],
                                      format="%Y%m%d",
                                      errors="coerce")
                if pd.notna(date):
                    orig_subs.append({
                        "date": date,
                        "submission": sub,
                        "record": rec
                    })

    if not orig_subs:
        return None

    full_orig_records = []

    for orig_record in orig_subs:
    
        #first = min(orig_subs, key=lambda x: x["date"])
        sub = orig_record["submission"]
        rec = orig_record["record"]
        openfda = rec.get("openfda", {})
        app_nr = rec.get("application_number")
        nct_ids_str, first_sentence, indications_text = fetch_application_label_details(app_nr)
        
        full_orig_records.append({
        "generic_name": drug,
        "brand_name": openfda.get("brand_name"),
        "sponsor_name": rec.get("sponsor_name"),
        "application_number": app_nr,
        "submission_type": sub.get("submission_type"),
        "submission_number": sub.get("submission_number"),
        "approval_date": sub.get("submission_status_date"),
        "nct_ids_str": nct_ids_str,
        "indications_label_first_sent": first_sentence,
        "indications_label_full_text": indications_text,
        "indication": rec.get("products", [{}])[0].get("indication"),
        "pharm_class_cs": openfda.get("pharm_class_cs"),
        "pharm_class_epc": openfda.get("pharm_class_epc"),
        "pharm_class_pe": openfda.get("pharm_class_pe"),
        "pharm_class_moa": openfda.get("pharm_class_moa")
    })

    return full_orig_records


# -----------------------------
# Core processing
# -----------------------------
def process_drugs_with_checkpoints(approved_index,
                                   processed,
                                   checkpoint_every,
                                   output_path,
                                   df_existing):
    """
    Iterate through (canonical_drug_name, drug_query_string),
    fetch *all* FDA approval entries (list of dicts), and write checkpoints.
    """
    records = []
    tic = time.time()

    for canonical_drug_name, drug in tqdm(approved_index, desc="Processing drugs"):
        if canonical_drug_name in processed:
            continue  # skip already processed drug name

        details_list = all_fda_approval_details(drug)

        # fallback heuristic (firstname/lastname swap) only if no results
        if not details_list:
            parts = str(drug).split()
            if len(parts) == 2:
                details_list = all_fda_approval_details(f"{parts[1]} {parts[0]}")

        # If still nothing, record a placeholder row (optional)
        if not details_list:
            records.append({
                "canonical_drug_name": canonical_drug_name,
                "queried_name": drug,
                "approval_year": None,
                "pharm_class_epc": None,
                "pharm_class_moa": None,
                "sponsor_name": None,
                "application_number": None,
                "brand_name": None,
                "indications_label_first_sent": None,
                "indications_label_full_text": None,
                "nct_ids_str": None,
            })
        else:
            # Iterate entries (list of dicts) → one output row per entry
            for d in details_list:
                # approval_date is like 'YYYYMMDD'
                approval_dt = pd.to_datetime(d.get("approval_date"), format="%Y%m%d", errors="coerce")
                approval_year = int(approval_dt.year) if pd.notna(approval_dt) else None

                brand = d.get("brand_name")
                # brand_name can be a list; normalize to string
                if isinstance(brand, list):
                    brand = ", ".join([str(b) for b in brand if b])

                records.append({
                    "canonical_drug_name": canonical_drug_name,
                    "queried_name": drug,
                    "approval_year": approval_year,
                    "pharm_class_epc": d.get("pharm_class_epc"),
                    "pharm_class_moa": d.get("pharm_class_moa"),
                    "sponsor_name": d.get("sponsor_name"),
                    "application_number": d.get("application_number"),
                    "brand_name": brand,
                    "indications_label_first_sent": d.get("indications_label_first_sent"),
                    "indications_label_full_text": d.get("indications_label_full_text"),
                    "nct_ids_str": d.get("nct_ids_str"),
                })

        # ---- checkpoint every N appended rows ----
        if checkpoint_every and (len(records) % checkpoint_every == 0):
            df_new = pd.DataFrame(records)
            df_ckpt = pd.concat([df_existing, df_new], ignore_index=True)

            # de-dupe per (drug, application, submission)
            dedupe_keys = ["canonical_drug_name", "application_number"]
            existing_keys = [k for k in dedupe_keys if k in df_ckpt.columns]
            if existing_keys:
                df_ckpt.drop_duplicates(subset=existing_keys, keep="last", inplace=True)

            save_csv_atomic(df_ckpt, output_path)
            print(f"[INFO] Checkpoint saved ({len(df_ckpt)} total rows).")
            df_existing = df_ckpt.copy()

    # ---- final save ----
    df_new = pd.DataFrame(records)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    dedupe_keys = ["canonical_drug_name", "application_number"]
    existing_keys = [k for k in dedupe_keys if k in df_final.columns]
    if existing_keys:
        df_final.drop_duplicates(subset=existing_keys, keep="last", inplace=True)

    save_csv_atomic(df_final, output_path)

    elapsed = time.time() - tic
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"[INFO] Processing took {h}h {m}m {s}s ({elapsed:.2f}s total)")
    return records, df_final


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load/merge existing progress if any
    df_existing, processed = resume_from_checkpoint(args.output)

    # Build the list/index of drugs to process based on term frequency
    approved_index = load_terms_and_build_index(
        terms_path=args.terms,
        articles_col=args.articles_col,
        terms_col=args.terms_col,
        min_articles=args.min_articles,
    )

    # Process with periodic checkpoints
    records, df_existing = process_drugs_with_checkpoints(
        approved_index=approved_index,
        processed=processed,
        checkpoint_every=args.checkpoint_every,
        output_path=args.output,
        df_existing=df_existing,
    )

    # Final save (existing + new), deduped
    df_final = pd.concat([df_existing, pd.DataFrame(records)], ignore_index=True)
    df_final.drop_duplicates(subset=["canonical_drug_name", "application_number"], keep="last", inplace=True)
    save_csv_atomic(df_final, args.output)

    print(f"[DONE] Saved full table with {len(df_final)} rows to {args.output}")


if __name__ == "__main__":
    main()