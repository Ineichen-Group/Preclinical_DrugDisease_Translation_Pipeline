#!/usr/bin/env python3
# save as pubmed_first_author_country.py

import argparse
import csv
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import unicodedata
import re
from typing import Dict, List, Optional, Tuple
import pycountry
from tqdm import tqdm
tqdm.pandas()
import joblib
import spacy
import numpy as np

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200  # NCBI is fine with 200 ids per efetch call
PAUSE_SECONDS = 0.35  # be polite (<3 req/sec)

# NOTE: The models for geolocation inference can be downloaded from: https://github.com/leebr27/affiliation-geoinference.

# --- PubMed fetch & parse ------------------------------------------------------

def efetch_pubmed_xml(pmids: List[str], email: Optional[str]=None, api_key: Optional[str]=None) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    r = requests.get(EFETCH_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def parse_first_author_affiliations(xml_text: str) -> Dict[str, Optional[str]]:
    """
    Returns {pmid: first_author_affiliation_text or None}
    """
    result: Dict[str, Optional[str]] = {}
    root = ET.fromstring(xml_text)

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        if not pmid:
            continue

        # Find first Author
        first_author = None
        author_list = art.find(".//AuthorList")
        if author_list is not None:
            for a in author_list.findall("Author"):
                # Skip CollectiveName-only entries
                if a.find("CollectiveName") is not None:
                    continue
                first_author = a
                break

        aff_text = None
        if first_author is not None:
            # PubMed has AffiliationInfo/Affiliation; sometimes multiple
            aff_infos = first_author.findall(".//AffiliationInfo")
            if aff_infos:
                # prefer the first affiliation listed for the first author
                aff_el = aff_infos[0].find("Affiliation")
                if aff_el is not None and aff_el.text:
                    aff_text = aff_el.text.strip()

        result[pmid] = aff_text

    # Handle PubmedBookArticle (rare)
    for art in root.findall(".//PubmedBookArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        if not pmid:
            continue
        # As a fallback, try first Author in Book
        first_author = None
        author_list = art.find(".//AuthorList")
        if author_list is not None:
            for a in author_list.findall("Author"):
                if a.find("CollectiveName") is not None:
                    continue
                first_author = a
                break
        aff_text = None
        if first_author is not None:
            aff_infos = first_author.findall(".//AffiliationInfo")
            if aff_infos:
                aff_el = aff_infos[0].find("Affiliation")
                if aff_el is not None and aff_el.text:
                    aff_text = aff_el.text.strip()
        result[pmid] = aff_text

    return result

US_STATES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

STATE_PATTERN = re.compile(r"\b(" + "|".join(map(re.escape, US_STATES.keys())) + r")\b", flags=re.I)

def extract_country(geo):
    if not geo or pd.isna(geo):
        return None
    parts = str(geo).split(",")
    return parts[-1].strip() if parts else None

def infer_countries_batched(
    df: pd.DataFrame,
    text_col: str,
    nlp,                        # spaCy Language object
    vectorizer,                 # your loaded_vectorizer
    model,                      # your loaded_model
    batch_size: int = 256,
    n_process: int = 1,
    show_progress: bool = True,
) -> pd.Series:
    """
    For each row:
      - if text contains a GPE/LOC (via spaCy) or a US state name -> use model prediction
      - else -> 'unlabeled'
    Returns a Series aligned to df.index.
    """
    s = df[text_col].fillna("").astype(str)

    # 1) spaCy pass in batches (fast + streaming)
    loc_labels = {"GPE", "LOC"}
    has_loc_spacy = []
    pipe = nlp.pipe(s.tolist(), batch_size=batch_size, n_process=n_process)
    if show_progress:
        pipe = tqdm(pipe, total=len(s), desc="spaCy NER")

    for doc in pipe:
        found = any(ent.label_ in loc_labels for ent in doc.ents)
        has_loc_spacy.append(found)
    has_loc_spacy = np.array(has_loc_spacy, dtype=bool)

    # 2) US state mention (word-bounded)
    has_us_state = s.str.contains(STATE_PATTERN, na=False)

    # 3) Combined mask of rows we’ll actually classify
    has_location = has_loc_spacy | has_us_state.values

    # 4) Initialize all as 'unlabeled'
    out = pd.Series("unlabeled", index=df.index, dtype=object)

    # 5) Predict only where a location signal exists
    idx = df.index[has_location]
    if len(idx) > 0:
        X = vectorizer.transform(s.loc[idx].tolist())
        y_pred = model.predict(X)
        # if your model returns numpy dtype, cast to py objects (strings) if needed
        out.loc[idx] = pd.Series(y_pred, index=idx).astype(object)

    # Optional: quick stats
    n_unlabeled = int((out == "unlabeled").sum())
    print(f"Labeled: {len(out) - n_unlabeled} | Unlabeled: {n_unlabeled}")

    return out

# --- Main I/O ------------------------------------------------------------------
def process_csv(input_csv: str, output_csv: str, pmid_column: str, email: Optional[str], api_key: Optional[str]) -> None:
    df = pd.read_csv(input_csv, dtype={pmid_column: str})
    if pmid_column not in df.columns:
        raise ValueError(f"Input CSV must contain a '{pmid_column}' column")

    have_affiliation_col = "first_author_affiliation" in df.columns

    # If we don't have affiliations yet, fetch them for ALL rows and save an intermediate file
    if not have_affiliation_col:
        pmids = (
            df[pmid_column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        # Handle empty input gracefully
        if not pmids:
            df["first_author_affiliation"] = ""
            df["first_author_country"] = None
            # Save intermediate (empty affiliations) and final (same) for consistency
            df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
            return

        n_batches = (len(pmids) + BATCH_SIZE - 1) // BATCH_SIZE
        aff_map: Dict[str, Optional[str]] = {}

        with tqdm(total=n_batches, desc="Fetching PubMed XML (batches)", unit="batch", dynamic_ncols=True) as pbar:
            for i in range(0, len(pmids), BATCH_SIZE):
                batch = pmids[i:i+BATCH_SIZE]
                try:
                    xml_text = efetch_pubmed_xml(batch, email=email, api_key=api_key)
                    parsed = parse_first_author_affiliations(xml_text)
                    aff_map.update(parsed)
                except Exception as e:
                    for pm in batch:
                        aff_map.setdefault(pm, None)
                    tqdm.write(f"[WARN] Failed batch {i//BATCH_SIZE+1}: {e}")
                finally:
                    pbar.update(1)
                time.sleep(PAUSE_SECONDS)

        # Attach affiliations
        df["first_author_affiliation"] = df[pmid_column].astype(str).map(aff_map)

        # --- Save INTERMEDIATE results (affiliations only) ---
        df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        print("Using existing 'first_author_affiliation' column from input CSV.")

    # From here: always compute/overwrite country based on whatever affiliation column exists
    spacy_obj = spacy.load("en_core_web_sm")
    loaded_model = joblib.load('./data/affiliation-geoinference/geoinference_linearsvc_1mil.joblib.lzma')
    loaded_vectorizer = joblib.load('./data/affiliation-geoinference/geoinference_vectorizer_1mil.joblib.lzma')
    
    df["first_author_geolocation"] = infer_countries_batched(
        df,
        text_col="first_author_affiliation",
        nlp=spacy_obj,                         # e.g., spacy.load("en_core_web_sm")
        vectorizer=loaded_vectorizer,
        model=loaded_model,
        batch_size=256,
        n_process=1,                           # bump if you want multi-process
    )
    df["first_author_country"] = df["first_author_geolocation"].apply(extract_country)

    # --- Save FINAL results (with country) ---
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    parser = argparse.ArgumentParser(
        description="Append first author affiliation and country from PubMed to a CSV with PMIDs."
    )
    parser.add_argument("input_csv", help="Path to input CSV containing a 'pmid' column (or specify --pmid-column).")
    parser.add_argument("output_csv", help="Path to write the output CSV.")
    parser.add_argument("--pmid-column", default="pmid", help="Name of the PMID column (default: pmid).")
    parser.add_argument("--email", default=None, help="Contact email for NCBI E-utilities (recommended).")
    parser.add_argument("--api-key", default=None, help="NCBI API key (optional, increases rate limits).")

    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        pmid_column=args.pmid_column,
        email=args.email,
        api_key=args.api_key,
    )

if __name__ == "__main__":
    main()
