import os
import requests
import pandas as pd
from tqdm import tqdm
import unicodedata
import re
import argparse
import time

def first_fda_approval_details(drug):
    q = f'openfda.generic_name:"{drug}"'
    try:
        resp = requests.get(
            "https://api.fda.gov/drug/drugsfda.json",
            params={"search": q, "limit": 100}
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


# -----------------------------
# Core processing
# -----------------------------
def process_drugs_with_checkpoints(approved_index,
                                   processed,
                                   checkpoint_every,
                                   output_path,
                                   df_existing):
    """
    Iterate through the approved index, skip already processed items, fetch metadata,
    and periodically save a full checkpoint (existing + new records, de-duplicated).
    """
    records = []
    tic = time.time()

    # tqdm over the full index; each item is (canonical_drug_name, drug_query_string)
    for canonical_drug_name, drug in tqdm(approved_index, desc="Processing drugs"):
        if canonical_drug_name in processed:
            continue  # skip already processed

        # Query for first FDA approval metadata
        details = first_fda_approval_details(drug)

        # Try "Lastname Firstname" → "Firstname Lastname" swap for two-token names (fallback heuristic)
        if not details:
            parts = drug.split()
            if len(parts) == 2:
                details = first_fda_approval_details(f"{parts[1]} {parts[0]}")

        # Extract fields robustly even if details is None
        approval_date = details.get("approval_date") if details else None
        epc = details.get("pharm_class_epc") if details else None
        moa = details.get("pharm_class_moa") if details else None
        sponsor_name = details.get("sponsor_name") if details else None

        records.append({
            "canonical_drug_name": canonical_drug_name,
            "queried_name": drug,
            "approval_year": approval_date.year if approval_date else None,
            "pharm_class_epc": epc,
            "pharm_class_moa": moa,
            "sponsor_name": sponsor_name
        })

        # Periodic checkpoint of ALL progress so far (existing + new), deduped by canonical_drug_name
        if checkpoint_every and (len(records) % checkpoint_every == 0):
            df_new = pd.DataFrame(records)
            df_ckpt = pd.concat([df_existing, df_new], ignore_index=True)
            df_ckpt.drop_duplicates(subset=["canonical_drug_name"], keep="last", inplace=True)

            save_csv_atomic(df_ckpt, output_path)
            print(f"[INFO] Checkpoint saved ({len(df_ckpt)} total rows).")

            # Refresh existing baseline so next checkpoint is incremental from here
            df_existing = df_ckpt.copy()

    # Timing summary
    elapsed = time.time() - tic
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print(f"[INFO] Processing took {h}h {m}m {s}s ({elapsed:.2f}s total)")

    return records, df_existing


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
    df_final.drop_duplicates(subset=["canonical_drug_name"], keep="last", inplace=True)
    save_csv_atomic(df_final, args.output)

    print(f"[DONE] Saved full table with {len(df_final)} rows to {args.output}")


if __name__ == "__main__":
    main()