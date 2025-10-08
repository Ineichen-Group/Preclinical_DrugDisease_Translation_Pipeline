#!/usr/bin/env python3
"""
Fetch first-author affiliations from PubMed given PMIDs.

Reads PMIDs from a large JSONL file and processes them in robust batches.
Features:
- Streamed JSONL reading (memory efficient)
- Resumable (auto-detects existing partial CSV)
- Batch-based fetching with exponential backoff retries
- Periodic intermediate saves
- Progress bar with tqdm

Usage:
    python fetch_affiliations.py \
        --input /path/to/combined_materials_methods.jsonl \
        --output affiliations_intermediate.csv \
        --email your_email@example.com \
        --api-key your_ncbi_api_key
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, Optional, List
import requests
import pandas as pd
from tqdm import tqdm
import argparse
import random
import xml.etree.ElementTree as ET

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

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

def parse_pubmed_authors_affiliations(xml_text: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Parse PubMed XML and extract:
      - first author's affiliation
      - full author list
      - all affiliations across all authors
    Returns:
        {pmid: {
            "first_author_affiliation": str or None,
            "authors": str (comma-separated),
            "all_affiliations": str (semicolon-separated)
        }}
    """
    results: Dict[str, Dict[str, Optional[str]]] = {}
    root = ET.fromstring(xml_text)

    for art_tag in [".//PubmedArticle", ".//PubmedBookArticle"]:
        for art in root.findall(art_tag):
            pmid_el = art.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
            if not pmid:
                continue

            author_list = art.find(".//AuthorList")
            authors = []
            all_affs = set()
            first_author_aff = None

            if author_list is not None:
                for idx, a in enumerate(author_list.findall("Author")):
                    # skip collective or corporate authors
                    if a.find("CollectiveName") is not None:
                        continue

                    last = a.findtext("LastName", "").strip()
                    fore = a.findtext("ForeName", "").strip()
                    name = f"{last} {fore}".strip()
                    if name:
                        authors.append(name)

                    # collect affiliations for this author
                    aff_infos = a.findall(".//AffiliationInfo")
                    for aff_info in aff_infos:
                        aff_el = aff_info.find("Affiliation")
                        if aff_el is not None and aff_el.text:
                            aff_text = aff_el.text.strip()
                            if aff_text:
                                all_affs.add(aff_text)
                                # only set first_author_aff on first author
                                if idx == 0 and first_author_aff is None:
                                    first_author_aff = aff_text

            results[pmid] = {
                "first_author_affiliation": first_author_aff,
                "authors": ", ".join(authors) if authors else None,
                "all_affiliations": "; ".join(sorted(all_affs)) if all_affs else None
            }

    return results

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_api_key_from_file(file_path: Path) -> str:
    """Read NCBI_API_KEY from a file with a line like 'NCBI_API_KEY=xxxx'."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NCBI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise ValueError(f"NCBI_API_KEY not found in {file_path}")

def load_pmids(jsonl_path: Path) -> List[str]:
    """Stream through JSONL and collect PMIDs (memory-efficient)."""
    pmids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "PMID" in obj:
                    pmids.append(str(obj["PMID"]))
            except json.JSONDecodeError:
                tqdm.write("[WARN] Skipping malformed JSON line.")
    return pmids


def save_partial_results(df: pd.DataFrame, output_path: Path):
    """Save partial affiliation map to CSV."""
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def fetch_with_retry(batch: List[str],
                     email: str,
                     api_key: str,
                     max_retries: int = 3,
                     base_delay: float = 2.0):
    """Try fetching PubMed XML with exponential backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            xml_text = efetch_pubmed_xml(batch, email=email, api_key=api_key)
            return xml_text
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            tqdm.write(f"[WARN] Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    return None


def robust_fetch_affiliations(pmids: List[str],
                              email: str,
                              api_key: str,
                              output_csv: Path,
                              batch_size: int,
                              pause_seconds: float,
                              max_retries: int,
                              base_retry_delay: float) -> pd.DataFrame:
    """Fetch first-author affiliations + full author list + all affiliations with retry and checkpointing."""
    n_batches = (len(pmids) + batch_size - 1) // batch_size
    data_map: Dict[str, Dict[str, Optional[str]]] = {}

    # Resume support
    if output_csv.exists():
        tqdm.write(f"[INFO] Loading existing partial results from {output_csv}")
        existing = pd.read_csv(output_csv, dtype=str)
        data_map.update({
            row["PMID"]: {
                "first_author_affiliation": row.get("first_author_affiliation"),
                "authors": row.get("authors"),
                "all_affiliations": row.get("all_affiliations")
            } for _, row in existing.iterrows()
        })
        pmids = [pm for pm in pmids if pm not in data_map]
        tqdm.write(f"[INFO] Resuming with {len(pmids)} remaining PMIDs.")

    with tqdm(total=n_batches, desc="Fetching PubMed XML (batches)", unit="batch", dynamic_ncols=True) as pbar:
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            try:
                xml_text = fetch_with_retry(
                    batch=batch,
                    email=email,
                    api_key=api_key,
                    max_retries=max_retries,
                    base_delay=base_retry_delay
                )
                parsed = parse_pubmed_authors_affiliations(xml_text)
                data_map.update(parsed)
            except Exception as e:
                tqdm.write(f"[ERROR] Failed batch {i // batch_size + 1} after retries: {e}")
                for pm in batch:
                    data_map.setdefault(pm, {
                        "first_author_affiliation": None,
                        "authors": None,
                        "all_affiliations": None
                    })
            finally:
                pbar.update(1)
                time.sleep(pause_seconds)

            # periodic checkpoint save every 10 batches
            if (i // batch_size + 1) % 10 == 0:
                tqdm.write("[INFO] Saving intermediate results...")
                df_partial = pd.DataFrame.from_dict(data_map, orient="index").reset_index(names="PMID")
                df_partial.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    # Final save
    df = pd.DataFrame.from_dict(data_map, orient="index").reset_index(names="PMID")
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return df


# ----------------------------
# MAIN EXECUTION
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch first-author affiliations from PubMed using PMIDs from a JSONL file."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output CSV file.")
    parser.add_argument("--email", required=True, type=str, default="simona.doneva@uzh.ch", help="Email address for NCBI API.")
    parser.add_argument("--api-key-file", required=True, type=Path,
                        help="Path to text file containing 'NCBI_API_KEY=your_key'.")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of PMIDs per batch (default: 200).")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds to pause between batches (default: 0.5).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed batches (default: 3).")
    parser.add_argument("--base-retry-delay", type=float, default=2.0, help="Base delay (seconds) for backoff (default: 2.0).")

    args = parser.parse_args()

    tqdm.write(f"[INFO] Reading PMIDs from {args.input}")
    pmids = load_pmids(args.input)
    tqdm.write(f"[INFO] Total PMIDs loaded: {len(pmids)}")
    
    api_key = load_api_key_from_file(args.api_key_file)

    df = robust_fetch_affiliations(
        pmids=pmids,
        email=args.email,
        api_key=api_key,
        output_csv=args.output,
        batch_size=args.batch_size,
        pause_seconds=args.pause,
        max_retries=args.max_retries,
        base_retry_delay=args.base_retry_delay
    )

    tqdm.write(f"[DONE] Saved final results to {args.output}")


if __name__ == "__main__":
    main()
