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

def strip_accents(s: str) -> str:
    """Strip accents/diacritics and return plain ASCII letters."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def norm_text(s: str) -> str:
    """Normalize: lowercase, strip accents, collapse spaces."""
    s = strip_accents(s or "")
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_country_index() -> Dict[str, str]:
    idx: Dict[str, str] = {}

    def put(key: str, val: str):
        """Add key to index in normalized form + ASCII fallback if needed."""
        if not key:
            return
        norm_key = norm_text(key)
        idx[norm_key] = val
        # If key had accents, also add plain ASCII fallback
        ascii_key = strip_accents(key).lower().strip()
        if ascii_key and ascii_key != norm_key:
            idx[ascii_key] = val

    # 1) From pycountry: official, common, alpha codes
    for c in pycountry.countries:
        canonical = c.name
        put(c.name, canonical)
        if getattr(c, "official_name", None):
            put(c.official_name, canonical)
        if getattr(c, "common_name", None):
            put(c.common_name, canonical)
        if getattr(c, "alpha_2", None):
            put(c.alpha_2, canonical)
        if getattr(c, "alpha_3", None):
            put(c.alpha_3, canonical)

    # 2) Common aliases & tricky cases
    aliases = {
        # USA
        "USA": "United States",
        "U.S.A": "United States",
        "U.S.A.": "United States",
        "US": "United States",
        "U.S.": "United States",
        "United States of America": "United States",
        # UK + home nations
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "England": "United Kingdom",
        "Scotland": "United Kingdom",
        "Wales": "United Kingdom",
        "Northern Ireland": "United Kingdom",
        # Netherlands
        "Netherlands": "Netherlands",
        "The Netherlands": "Netherlands",
        # Czechia
        "Czech Republic": "Czechia",
        # Russia
        "Russian Federation": "Russia",
        "Russia": "Russia",
        # Korea
        "South Korea": "Korea, Republic of",
        "Republic of Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Republic of",
        # Ivory Coast
        "Ivory Coast": "Côte d'Ivoire",
        # Eswatini
        "Swaziland": "Eswatini",
        # Turkey
        "Türkiye": "Turkey",
        "Turkey": "Turkey",
        # Hong Kong, Macao
        "Hong Kong": "China",
        "Macau": "China",
        "Macao": "China",
        # Taiwan
        "Taiwan": "Taiwan",
        # Others
        "Vatican City": "Holy See",
        "Bolivia": "Bolivia, Plurinational State of",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Moldova": "Moldova, Republic of",
        "Syria": "Syrian Arab Republic",
        "Laos": "Lao People's Democratic Republic",
        "Brunei": "Brunei Darussalam",
        "Cape Verde": "Cabo Verde",
        "Palestine": "Palestine, State of",
        "Iran": "Iran, Islamic Republic of",
        "Vietnam": "Viet Nam",
        "East Timor": "Timor-Leste",
        "Micronesia": "Micronesia, Federated States of",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Saint Kitts and Nevis": "Saint Kitts and Nevis",
    }
    for k, v in aliases.items():
        put(k, v)

    # 3) Local language forms
    local = {
        "Deutschland": "Germany",
        "Österreich": "Austria",
        "Suisse": "Switzerland",
        "Schweiz": "Switzerland",
        "Svizzerra": "Switzerland",
        "España": "Spain",
        "Brasil": "Brazil",
        "Norge": "Norway",
        "Sverige": "Sweden",
        "Suomi": "Finland",
        "Danmark": "Denmark",
        "Ελλάδα": "Greece",
        "Hellas": "Greece",
        "日本": "Japan",
        "Nippon": "Japan",
        "中国": "China",
        "Praha, Česká republika": "Czechia",
    }
    for k, v in local.items():
        put(k, v)

    return idx

# Build index + word set
COUNTRY_INDEX = build_country_index()
COUNTRY_WORDSET = set(COUNTRY_INDEX.keys())
ASCII_EXTRAS = {"turkey": "Turkey"}  # extend as needed
COUNTRY_WORDSET_WITH_ASCII = set(COUNTRY_WORDSET) | set(ASCII_EXTRAS.keys())

COUNTRY_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, COUNTRY_WORDSET_WITH_ASCII), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE
)

def extract_country(geo):
    if not geo or pd.isna(geo):
        return None
    parts = str(geo).split(",")
    return parts[-1].strip() if parts else None

def has_country_keyword(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return COUNTRY_PATTERN.search(text) is not None

def has_us_state(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return STATE_PATTERN.search(text) is not None

def spacy_gpe_flags(series: pd.Series, nlp, batch_size=256, n_process=1, show_progress=True) -> np.ndarray:
    loc_labels = {"GPE", "LOC"}
    texts = series.fillna("").astype(str).tolist()
    pipe = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    if show_progress:
        pipe = tqdm(pipe, total=len(texts), desc="spaCy NER")
    flags = []
    for doc in pipe:
        flags.append(any(ent.label_ in loc_labels for ent in doc.ents))
    return np.array(flags, dtype=bool)

def infer_geolocation_batched(
    df: pd.DataFrame,
    text_col: str,
    nlp,                    # spaCy Language object (or None to disable spaCy)
    vectorizer,             # loaded TfidfVectorizer (or Pipeline)
    model,                  # loaded classifier (if vectorizer is separate); if Pipeline, pass it here and set vectorizer=None
    batch_size: int = 256,
    n_process: int = 1,
    show_progress: bool = True,
) -> pd.Series:
    """
    Returns a Series 'first_author_geolocation':
      - full predicted string from the model where any trigger hits (spaCy GPE/LOC OR country keyword OR US state)
      - 'unlabeled' otherwise
    """
    s = df[text_col].fillna("").astype(str)

    # Triggers
    trig_spacy = spacy_gpe_flags(s, nlp, batch_size, n_process, show_progress) if nlp else np.zeros(len(s), bool)
    trig_country = s.apply(has_country_keyword).to_numpy()
    trig_state = s.apply(has_us_state).to_numpy()
    mask = trig_spacy | trig_country | trig_state

    out = pd.Series("unlabeled", index=df.index, dtype=object)

    idx = df.index[mask]
    if len(idx) > 0:
        texts = s.loc[idx].tolist()
        if vectorizer is not None:
            X = vectorizer.transform(texts)
            y_pred = model.predict(X)
        else:
            # model is a Pipeline (Vectorizer+Classifier)
            y_pred = model.predict(texts)
        out.loc[idx] = pd.Series(y_pred, index=idx).astype(object)

    # Optional: quick stats
    print(f"Triggered rows: {mask.sum()} / {len(mask)} | Unlabeled: {(out == 'unlabeled').sum()}")
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
    
    df["first_author_geolocation"] = infer_geolocation_batched(
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
