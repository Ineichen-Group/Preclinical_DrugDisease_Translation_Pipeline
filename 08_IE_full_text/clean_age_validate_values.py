import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import re
import time
from tqdm import tqdm
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import spacy
from typing import Dict, List
import os
tqdm.pandas()  # enables progress_apply on DataFrames
import argparse
from regex_classifiers.species_classifier import SpeciesClassifier
from num2words import num2words

# Load SciSpacy model
def load_nlp_model():
    return spacy.load("en_core_sci_sm")

# Common regex
age_range_re = re.compile(r"\d+\s*[-–]\s*\d+", flags=re.IGNORECASE)
week_re = re.compile(r"(\d+(?:\.\d+)?)\s*(weeks?|wks?)", flags=re.IGNORECASE)
species_clf = SpeciesClassifier()

# Compiled regex pattern for week-related age expressions
age_keywords_pattern = r"""
\b(
    age|ages|aged|aging|old|older|young|adult|adults|senescent|
    neonatal|neonate|newborn|pup|pups|juvenile|weanling|weaning|
    postnatal|prenatal|prepubescent|fetal|fetus|fetuses|
    week[-\s]?old|weekold|                         # forms like "week-old", "weekold"
    \d+\s*(?:-|–|to)\s*\d+\s*(week|wk)s?\b|             # range: e.g., "6–8 weeks"
    \d+\s*(week|wk)s?(?!\s*old)|                   # standalone: e.g., "8 weeks" but not "8 weeks old"
    after\s+birth|post[-\s]?birth
)\b
"""

# Compile the pattern once
_age_week_regex = re.compile(age_keywords_pattern, flags=re.IGNORECASE | re.VERBOSE)

def contains_week_age_expression(text: str) -> bool:
    """
    Returns True if the text contains a week-related age expression, else False.
    """
    return bool(_age_week_regex.search(text))

def contains_animal_expression(text: str) -> bool:
    """
    Returns True if the text contains an animal-related expression, else False.
    """
    _, found_labels = species_clf.classify(text)

    if len(found_labels) == 1 and found_labels[0] == "species-other":
        if not re.search(r'\banimals?\b', text, flags=re.IGNORECASE):
            return False
        
    return True

def extract_weekly_ages(df: pd.DataFrame,
                        label_col: str = 'prediction_encoded_label',
                        raw_col: str   = 'age_prediction') -> pd.DataFrame:
    """
    From a doc‑level DataFrame, keep only entries:
      - whose encoded label is NOT a range (e.g. “3-4 weeks”)
      - whose raw prediction mentions “weeks”
    Then explode comma‑separated labels, strip whitespace,
    and pull out numeric + unit into separate columns.
    """
    # 1) Drop any with hyphen/en‑dash ranges in the encoded label
    mask_not_range = ~df[label_col].astype(str).str.contains(r'\d+\s*[-–]\s*\d+')
    #    AND keep only those where the RAW prediction mentions “weeks”
    mask_has_weeks = df[raw_col].str.contains('weeks', case=False, na=False)
    df2 = df[mask_not_range & mask_has_weeks].copy()

    # 2) Explode comma‑separated age strings
    df2[label_col] = df2[label_col].fillna('')
    df2 = df2.assign(
        age_str = df2[label_col].str.split(',')
    ).explode('age_str')

    # 3) Clean up whitespace
    df2['age_str'] = df2['age_str'].str.strip()

    # 4) Split into numeric & unit
    split = df2['age_str'].str.split(r'\s+', n=1, expand=True)
    df2['age_num']  = pd.to_numeric(split[0], errors='coerce')
    df2['age_unit'] = split[1].str.lower().fillna('')

    # Optional: drop any rows where age_num is NaN
    df2 = df2[df2['age_num'].notna()].reset_index(drop=True)
    return df2

# Helper: normalize simple too-high ages into plausible ranges
def normalize_age(age_str: str) -> str:
    """
    Turn something like '200 weeks' → '2-00 weeks' or
    '350 weeks' → '3-50 weeks' by inserting a plausible hyphen split.
    """
    text = age_str.replace("weeks old", "weeks")
    parts = text.split()
    if len(parts) != 2:
        return age_str
    age, unit = parts
    if "-" not in age:
        try:
            val = float(age)
            if val > 150:
                s = age
                # e.g. '200' → '2-00', '350' → '3-50'
                if len(s) == 3:
                    age = f"{s[0]}-{s[1:]}"
                else:
                    age = f"{s[:2]}-{s[2:]}"
        except ValueError:
            pass
    return f"{age} {unit}"


def clean_too_high(df_flat: pd.DataFrame) -> dict:
    """
    Identify any rows where age_num > 150 weeks,
    produce a mapping { PMID: [ { old_str: new_str }, … ] }.
    """
    df = df_flat.copy()
    # Only look at rows we know are 'weeks'
    df = df[df['age_unit'].str.contains('weeks', case=False, na=False)]
    # Build the mapping
    mapping = defaultdict(list)
    for _, row in df[df['age_num'] > 150].iterrows():
        old = row['age_str']
        new = normalize_age(old)
        mapping[row['PMID']].append({old: new})
    return dict(mapping)

def resolve_age_from_text(age_text_to_check: str, current_age: str, current_age_time: str) -> str:
    """
    Attempts to validate or correct a predicted age based on context from biomedical text.

    - If exact age/unit is found, returns it as is.
    - If age looks like a misparsed range (e.g., '58' from '5–8'), tries to recover.
    - If the context includes 'P56'-style notation, overrides the unit to 'days'.
    - Otherwise, returns the original prediction.
    """
    current_age = str(current_age).strip()
    current_age_time = str(current_age_time).strip().lower()

    # Step 1: Look for exact age/unit
    exact_patterns = [
        rf'\b{current_age}\b(?!\s*[%\w])',
        rf'\b{current_age}\s*(weeks?|wks?|months?|mos?|days?)\b(?!\s*%)',
        rf'\b{current_age}[-\s]?(weeks?|wks?|months?|mos?|days?)[-\s]?old\b(?!\s*%)',
    ]
    for pattern in exact_patterns:
        if re.search(pattern, age_text_to_check, flags=re.IGNORECASE):
            return f"{current_age} {current_age_time}"

    # Step 2: Try to recover from a range like "5–8" misparsed as "58"
    if len(current_age) == 2 and current_age.isdigit():
        a, b = current_age[0], current_age[1]
        range_patterns = [
            rf'\b{a}\s*(to|-|–|—)\s*{b}\s*(weeks?|wks?|months?|mos?|days?)?\b',
            rf'\b{a}-{b}\s*(weeks?|wks?|months?|mos?|days?)?\b',
            rf'\b{a}–{b}\s*(weeks?|wks?|months?|mos?|days?)?\b',
        ]
        for pattern in range_patterns:
            if re.search(pattern, age_text_to_check, flags=re.IGNORECASE):
                return f"{a}-{b} {current_age_time}"

    # Step 3: Override unit if context includes P56-style notation
    p_matches = re.findall(r'\bP(\d{1,3})(?=[)\s,\.])', age_text_to_check)
    if current_age in p_matches:
        print(f"Overriding unit to 'days' based on context match with P{current_age}")
        print(f"[DONE] Resolved age from text: {current_age} days")
        return f"{current_age} days"

    # Step 4: Salvage likely real age from high predicted age (e.g. Fifty seven-week-old → 7 weeks and not 57 weeks)
    if current_age.isdigit():
        for digit_char in current_age:
            digit = int(digit_char)
            if digit == 0:
                continue  # skip zero
            word = num2words(digit)  # e.g., "8" → "eight"
            pattern = rf'\b{word}-week-old\b'
            if re.search(pattern, age_text_to_check, flags=re.IGNORECASE):
                print(f"Matched digit-derived age: {digit} from '{word}-week-old' in text")
                print(f"[DONE] Resolved age from text: {digit} {current_age_time}")
                return f"{digit} {current_age_time}"

    # Step 6: Fallback
    print(f"[DONE] Resolved age from text: {current_age} {current_age_time}")
    return f"{current_age} {current_age_time}"


# Stage 2: Clean via PMC
EMAIL = "youremail@example.com"
def pmid_to_pmcid(pmid: str) -> str:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=age-cleaner&email={EMAIL}&ids={pmid}&format=json"
    data = requests.get(url).json()
    rec = data.get('records', [{}])[0]
    return rec.get('pmcid')

_age_regex = re.compile(r"week[-\s]?old|\d+[-–]\d+\s*weeks?|\d+\s*weeks?", flags=re.IGNORECASE)

def contains_week_age(text: str) -> bool:
    return bool(_age_regex.search(text))


def normalize_dashes(text: str) -> str:
    # Replace en dash (–), em dash (—), and minus (−) with regular hyphen (-)
    return re.sub(r"[–—−]", "-", text)

def clean_via_pmc(df_flat, pmcid_file: str = None, map1=None,):
    """
    Look up medium‐sized week values in PMC, but skip any (PMID, age_str)
    pairs already corrected in map1.
    """
    # Build an exclusion set from map1 if provided
    exclude = set()
    if map1:
        for pmid, lst in map1.items():
            for d in lst:
                old = next(iter(d.keys()))
                exclude.add((pmid, old))

    mapping = defaultdict(list)
    df = df_flat.copy()
    df['flat'] = df['age_str']
    df[['num', 'unit']] = df['flat'].str.extract(r"(\d+)\s*(\w+)")
    df['num'] = pd.to_numeric(df['num'], errors='coerce')
    
    # Only those ~40–150 weeks
    to_check = df[(df['num'] > 40) & (df['num'] < 150)].copy()

    # Skip anything in stage‑1 corrections
    to_check = to_check[~to_check.apply(
        lambda r: (r['PMID'], r['flat']) in exclude,
        axis=1
    )]

    # PMC lookup
    nlp = spacy.load("en_core_sci_sm")
    
    # Load or lookup PMCIDs
    if pmcid_file and os.path.isfile(pmcid_file):
        pmc_df = pd.read_csv(pmcid_file, usecols=['PMID','PMCID'])
        pmc_df = pmc_df[pmc_df['PMCID'].notna()].copy()
        pmc_map = dict(zip(pmc_df['PMID'], pmc_df['PMCID']))
        to_check['PMCID'] = to_check['PMID'].map(pmc_map)
    else:
        tqdm.pandas()
        print("PMCID file not found. Fetching PMCID data instead...")
        to_check['PMCID'] = to_check['PMID'].progress_apply(pmid_to_pmcid)

    has = to_check[to_check['PMCID'].notna()]

    for _, row in tqdm(has.iterrows(), total=len(has)):
        pmc  = row['PMCID']
        age  = int(row['num'])
        unit = row['unit']

        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        seen = set()
        age_sentences = []
        for section in soup.find_all("section"):
            heading = section.find(["h2","h3","h4"])
            htxt    = heading.get_text(strip=True) if heading else ""
            if htxt in {"Abstract","Results","Discussion","Abbreviations"}:
                continue
            for p in section.find_all("p"):
                doc = nlp(p.get_text(strip=True))
                for sent in doc.sents:
                    stxt = sent.text.strip()
                    if contains_week_age_expression(stxt) and contains_animal_expression(stxt) and (stxt not in seen):
                        seen.add(stxt)
                        age_sentences.append(stxt)
                    
        age_text = normalize_dashes(" ".join(age_sentences))
        print(f"Processing PMID {row['PMID']} PMCID {pmc} → {age_text}, age={age}, unit={unit}")
        corrected = resolve_age_from_text(age_text, age, unit)
        mapping[row['PMID']].append({row['flat']: corrected})

    return dict(mapping), to_check

# Stage 3: Clean via Publisher Direct Access

def starts_with_cap_block(text, min_length=20):
    """
    Checks if the text starts with a block of ≥ `min_length` consecutive uppercase letters.

    Args:
        text (str): The input text.
        min_length (int): Minimum number of capital letters to consider a match.

    Returns:
        bool: True if it starts with ≥ `min_length` uppercase letters, False otherwise.
    """
    match = re.match(rf'^[A-Z]{{{min_length},}}', text.strip())
    return bool(match)

def is_reference_line(text):
    """
    Determines whether a line resembles a citation or reference entry.

    Args:
        text (str): A line or text chunk.

    Returns:
        bool: True if it's likely a reference, False otherwise.
    """
    text = text.strip()

    # Common: Number + year + page numbers (e.g., "825 1999 189 193")
    if re.search(r'\b\d{3,4}\s+\d{4}\s+\d{2,4}\s+\d{2,4}\b', text):
        return True

    # Contains "et al." + year
    if re.search(r'\bet al\.,?\s+\d{4}', text):
        return True

    # 2 or more people with initials, e.g., "J.L. Ferrara P. Morell"
    if len(re.findall(r'\b[A-Z]\.[A-Z]?\.?\s+[A-Z][a-z]+', text)) >= 2:
        return True

    # Journal name or abbreviation at the end (e.g., "Ann.")
    if re.search(r'\b[A-Z][a-z]{1,}\.$', text):
        return True

    # Starts with a number and has a comma-year author pattern
    if re.match(r'\d{3,4}', text) and re.search(r'[A-Z][a-z]+,\s*\d{4}', text):
        return True

    return False

def is_numeric_metadata_line(text):
    """
    Detects lines that are likely metadata or serial data (mostly numbers).

    Args:
        text (str): Input line or chunk.

    Returns:
        bool: True if it's mostly numeric/ID-like, False if not.
    """
    text = text.strip()
    tokens = text.split()

    # Case: starts with "serial" or "JL", followed by mostly digits
    if tokens and tokens[0].lower() in {"serial", "jl"}:
        return True

    # If more than 80% of tokens are numbers
    numeric_tokens = sum(t.isdigit() for t in tokens)
    if len(tokens) > 0 and numeric_tokens / len(tokens) > 0.8:
        return True

    # If there are 5+ number tokens and no verbs or sentence structure
    if numeric_tokens >= 5:
        return True

    return False

def pmid_to_doi(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching PMID {pmid}")
        return None

    root = ET.fromstring(response.content)
    for article_id in root.findall(".//ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            return article_id.text
    return None

def doi_to_html_url(doi):
    return f"https://doi.org/{doi}"  # Redirects to publisher page

def get_elsevier_full_text(pii, api_key):
    """
    Fetches the full article text from Elsevier API using the given PII and API key.

    Args:
        pii (str): Publisher Item Identifier for the article.
        api_key (str): Your Elsevier API key.

    Returns:
        str: Full article text if successful, otherwise fallback message.
    """
    url = f"https://api.elsevier.com/content/article/pii/{pii}"
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }

    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None

    try:
        data = res.json()
        return data['full-text-retrieval-response']['originalText']
    except Exception:
        return None
    
def extract_springer_article_text(html_text):
    """
    Extracts the main article content from Springer HTML page.

    Args:
        html_text (str): The raw HTML content of the Springer article page.

    Returns:
        str or None: Cleaned article text if found, otherwise None.
    """
    soup = BeautifulSoup(html_text, 'html.parser')
    article_div = soup.find('div', {'class': 'c-article-body'})
    
    if article_div:
        return article_div.get_text(separator='\n', strip=True)
    
    return None

def get_content_with_selenium(url: str, wait_time: int = 5) -> str:
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        time.sleep(wait_time)  # Let JS challenge resolve
        page_source = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    article_div = soup.find('div', class_='article-body')  # or whatever the body class is
    
    if article_div:
        text = article_div.get_text(separator="\n", strip=True)
        return text
    else:
        return None
    
def get_normalized_age_from_publisher(pmid: str, current_age: str, current_age_time: str, api_key: str) -> str:
    """
    Retrieves article text from Elsevier API using PMID, finds age-related sentences,
    and returns a normalized age label like "8-10 week".
    """
   
    # Step 1: Get DOI
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DOI-fetcher/1.0; donevasimona@gmail.com)"
    }
    time.sleep(1.5)  # Delay for NCBI API
    doi = pmid_to_doi(pmid)
    if not doi:
        print(f"no doi found for {pmid}")
        return f"{current_age} {current_age_time}"

    # Step 2: Get ScienceDirect redirect URL
    response = requests.get(doi_to_html_url(doi), headers=headers, allow_redirects=True)
    sciencedirect_url = response.url

    # Step 3: Extract PII if possible
    match = re.search(r'/pii/([A-Z0-9]+)', sciencedirect_url)
    
    if not match:
        # Step 4: Get full text directly
        print(f"processing {pmid} with bsoup")
        original_text = extract_springer_article_text(response.text)

        # try with selenium 
        if not original_text:
            print(f"processing {pmid} with selenium {response.url}")
            original_text = get_content_with_selenium(response.url)    
    else:
        # Step 4: Get full text from Elsevier API
        pii = match.group(1)
        print(f"processing  {pmid} with Elsevier {pii}")
        original_text = get_elsevier_full_text(pii, api_key)
        
    if not original_text:
        print(f"no original text found for: {pmid}")
        return f"{current_age} {current_age_time}"

    # Step 5: Extract and classify age-related sentences
    #print(original_text)
    nlp = spacy.load("en_core_sci_sm")
    doc = nlp(original_text)
    age_sentences = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if starts_with_cap_block(sentence_text):
            continue
        if is_numeric_metadata_line(sentence_text):
            continue
        #label, _ = age_clf.classify(sentence_text)
        if contains_week_age_expression(sentence_text) and contains_animal_expression(sentence_text):
            age_sentences.append(sentence_text)

    # Step 6: Normalize and return
    age_text_to_check = normalize_dashes(' '.join(age_sentences))
    #print(age_text_to_check)
    return resolve_age_from_text(age_text_to_check, current_age, current_age_time)
    

# Stage 4: Apply to original predictions
def apply_mappings(df_orig: pd.DataFrame, *maps: dict) -> pd.DataFrame:
    merged = defaultdict(list)
    for m in maps:
        for pmid, lst in m.items(): merged[pmid].extend(lst)
    def mapper(row):
        labels = [l.strip() for l in row['prediction_encoded_label'].split(',')]
        new = []
        for lab in labels:
            replaced = False
            for mm in merged.get(row['PMID'], []):
                if lab in mm:
                    new.append(mm[lab]); replaced = True; break
            if not replaced: new.append(lab)
        return ', '.join(new)
    df = df_orig.copy()
    df['verified_prediction_encoded_label'] = df.apply(mapper, axis=1)
    return df

def mapping_to_df(mapping: Dict[str, List[Dict[str, str]]]) -> pd.DataFrame:
    """
    Flattens a mapping of the form
      { PMID: [ {old_str: new_str}, … ], … }
    into a DataFrame with columns: PMID, old_age_str, new_age_str
    """
    records = []
    for pmid, corrections in mapping.items():
        for corr in corrections:
            # each corr is a dict of length 1
            old_str, new_str = next(iter(corr.items()))
            records.append({
                "PMID": pmid,
                "old_age_str": old_str,
                "new_age_str": new_str
            })
    return pd.DataFrame.from_records(records)

def load_elsevier_api_key(filepath="api_keys.txt"):
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("elsevier_api_key_uoz="):
                return line.strip().split("=")[1].strip('"')
    raise ValueError("API key not found in file.")

def create_pmid_mapping(df: pd.DataFrame, pmid_col: str = "PMID", flat_col: str = "prediction_encoded_label_flat", new_col: str = "prediction_encoded_label_new") -> dict:
    """
    Creates a mapping from PMID to a list of dictionaries where each dictionary maps
    the flat prediction to the new prediction.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with at least three specified columns.
    - pmid_col (str): Column name containing the PMID.
    - flat_col (str): Column name containing the flat label.
    - new_col (str): Column name containing the new label.

    Returns:
    - dict: A defaultdict(list) mapping each PMID to a list of {flat: new} dictionaries.
    """
    mapping = defaultdict(list)
    for _, row in df.iterrows():
        pmid = row[pmid_col]
        flat = row[flat_col]
        new = row[new_col]
        mapping[pmid].append({flat: new})
    return mapping


if __name__ == '__main__':
    # Load original
    #df_orig = pd.read_csv("./model_predictions/age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions.csv")
    
    parser = argparse.ArgumentParser(description="NER CSV prediction processor")
    parser.add_argument(
        "--input_csv",
        default="./model_predictions/age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions.csv",
        help="Path to the input CSV file containing predictions. Default is age_unsloth_meta_llama_3.1_8b."
    )
    args = parser.parse_args()

    # Load the input CSV from the argument
    df_orig = pd.read_csv(args.input_csv)
    
    print(f"Loaded {len(df_orig)} original age predictions.")
    # Preprocess flat data
    df_flat = extract_weekly_ages(df_orig)
    
    # Stage 1
    map1 = clean_too_high(df_flat)
    df_map1 = mapping_to_df(map1)
    save_path = "./model_predictions/age/post_processing/df_age_processed_too_high_values_20250722.csv"
    df_map1.to_csv(
        save_path,
        index=False
    )
    print(f"Stage 1 mappings saved to {save_path}")
    
    # Stage 2
    path_mapping_to_pmc = "./model_predictions/age/post_processing/df_incl_pmcid_20250722.csv"
    map2, df_with_pmcid = clean_via_pmc(df_flat, pmcid_file=path_mapping_to_pmc, map1=map1)
    df_map2 = mapping_to_df(map2)
    save_path2 = "./model_predictions/age/post_processing/df_age_processed_via_pmc_20250722.csv"
    
    df_map2.to_csv(
        save_path2,
        index=False
    )
    df_with_pmcid.to_csv(
        path_mapping_to_pmc,
        index=False)
    print(f"Stage 2 mappings saved to {save_path2}")
    
    # Stage 3 (need API key loaded)
    API_KEY = load_elsevier_api_key("../07_full_text_retrieval/api_keys.txt")
    save_path3 = "./model_predictions/age/post_processing/df_age_processed_via_publisher_20250722.csv"
    
    if save_path3 and os.path.isfile(save_path3):
        df_age_to_validate_no_pmc = pd.read_csv(save_path3)
        print(f"Loaded existing Stage 3 data from {save_path3}")
    else:
        print(f"Creating new Stage 3 data, saving to {save_path3}")
        df_age_to_validate_no_pmc = df_with_pmcid[~df_with_pmcid['PMCID'].notna()].copy()
        df_age_to_validate_no_pmc["prediction_encoded_label_new"] = df_age_to_validate_no_pmc.progress_apply(
            lambda row: get_normalized_age_from_publisher(
                pmid=row["PMID"],
                current_age=str(int(row["age_num"])),
                current_age_time=row["age_unit"],
                api_key=API_KEY
            ),
            axis=1
        )
        
        df_age_to_validate_no_pmc.to_csv(
            save_path3,
            index=False
        )
        print(f"Stage 3 mappings saved to {save_path3}")
    
    # Stage 4
    map3 = create_pmid_mapping(
        df_age_to_validate_no_pmc,
        pmid_col="PMID",
        flat_col="flat",
        new_col="prediction_encoded_label_new"
    )
    df_final = apply_mappings(df_orig, map1, map2, map3)  # add map3 when available
    save_path_final = "./model_predictions/age/df_age_predictions_verified_20250722.csv"
    df_final.to_csv(save_path_final, index=False)
    print(f"Cleaned predictions saved to {save_path_final}")
