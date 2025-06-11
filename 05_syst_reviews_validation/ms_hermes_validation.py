import pandas as pd
from time import sleep
from tqdm import tqdm
import os
from pubmed_utils import (
    get_pmids_for_title,
    get_pmids_for_doi,
    get_title_for_pmid,
    is_title_match,
)

INPUT_CSV = "05_syst_reviews_validation//ms_berger_sr/HERMES_INCLUDED.csv"       # Replace with your actual file
OUTPUT_CSV = "05_syst_reviews_validation/ms_berger_sr/HERMES_INCLUDED_PMIDs_valid_title.csv"

def process_dois(input_csv, output_csv, threshold=90):
    # Read and normalize header names to lowercase → original
    df = pd.read_csv(input_csv)
    col_map = {col.lower(): col for col in df.columns}
    
    # Ensure we have the columns we need
    needed = {'doi', 'title'}
    if not needed.issubset(col_map):
        missing = needed - set(col_map)
        raise ValueError(f"CSV must contain columns {missing} (case-insensitive)")
    
    doi_col   = col_map['doi']
    title_col = col_map['title']
    
    # Prepare (or load) the output file with fixed 'doi','pmid','title' headers
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        print(f"Found existing output CSV. Loading {len(df_existing)}...")
    else:
        df_existing = pd.DataFrame(columns=["doi", "pmid", "title"])
        df_existing.to_csv(output_csv, index=False)
    
    existing_dois   = set(df_existing['doi'].dropna())
    existing_titles = set(df_existing['title'].dropna())
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        doi   = row.get(doi_col)
        title = row.get(title_col)
        
        if pd.isna(title):
            continue
        
        # skip if already done
        if (pd.notna(doi) and doi in existing_dois) or title in existing_titles:
            continue
        
        # 1) Try via DOI
        pmids = get_pmids_for_doi(doi) if pd.notna(doi) else []
        matched = False
        for pmid in pmids:
            sleep(0.34)
            actual = get_title_for_pmid(pmid)
            if actual and is_title_match(title, actual, threshold):
                pd.DataFrame([{"doi": doi, "pmid": pmid, "title": title}]) \
                  .to_csv(output_csv, mode='a', header=False, index=False)
                existing_dois.add(doi)
                existing_titles.add(title)
                matched = True
                break
        
        # 2) Fall back to title search
        if not matched:
            print(f"No DOI hit for {doi!r}, searching by title…")
            title_pmids = get_pmids_for_title(title)
            if title_pmids:
                pd.DataFrame([{
                    "doi": doi if pd.notna(doi) else "",
                    "pmid": title_pmids[0],
                    "title": title
                }]).to_csv(output_csv, mode='a', header=False, index=False)
                existing_dois.add(doi)
                existing_titles.add(title)
            else:
                print(f"No PMIDs found for title '{title}'")
    
    print(f"Done — results in '{output_csv}'")
    
def main(fetch_dois=False):
    #print(get_pmids_for_doi(doi="10.1016/j.intimp.2018.12.001"))
    if fetch_dois:
        process_dois(INPUT_CSV, OUTPUT_CSV)
   
        
if __name__ == "__main__":
    main(fetch_dois=True)