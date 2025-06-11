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
    
def load_current_pmids():
    # All studies returned from the pubmed query
    all_original_pmids_file = "01_pubmed_query_neuro/pmids_with_data.txt"
    with open(all_original_pmids_file) as f:
        all_query_pmids = set(map(int, filter(str.isdigit, f.read().splitlines())))
        
    # Predicted animal studies
    all_animal_studies = pd.read_csv("02_animal_study_classification/model_predictions/all_animal_studies_clean_complete.csv")

    # All studies with drug and disease mention which will be kept for future processing
    identified_studies = pd.read_csv("03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_595768.csv")[['PMID', 'unique_interventions_linkbert_predictions', 'unique_conditions_linkbert_predictions']]

    print(f"Original PMIDs: {len(all_query_pmids)}")
    print(f"Animal studies PMIDs: {len(all_animal_studies)}")
    print(f"Identified studies PMIDs: {len(identified_studies)}")
    return all_query_pmids, all_animal_studies, identified_studies
    
def check_missing_pmids(hermes_df, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load current PMIDs
    all_query_pmids, all_animal_studies, identified_studies = load_current_pmids()

    # Full target set
    target_pmids = set(hermes_df['pmid'])
    print(f"Target PMIDs from HERMES-MS ds: {len(target_pmids)}")

    # 1. NER disease-specific filter
    disease = "multiple sclerosis"
    filtered = identified_studies[
        identified_studies['unique_conditions_linkbert_predictions']
        .str.lower()
        .str.contains(disease, case=False, na=False)
    ]
    disease_pmids = set(filtered['PMID'])
    missing_disease = target_pmids - disease_pmids
    print(f"Missing after disease-specific NER filter: {len(missing_disease)}")
    pd.DataFrame(missing_disease, columns=["pmid"]).to_csv(f"{output_dir}/missing_disease_filter.csv", index=False)

    # 2. Original query PMIDs
    original_pmids = set(all_query_pmids)
    missing_original = target_pmids - original_pmids
    print(f"Missing after original query filter: {len(missing_original)}")
    pd.DataFrame(missing_original, columns=["pmid"]).to_csv(f"{output_dir}/missing_original_query.csv", index=False)

    # 3. Animal studies PMIDs
    animal_pmids = set(all_animal_studies['PMID'])
    missing_animal = target_pmids - animal_pmids
    print(f"Missing after animal studies filter: {len(missing_animal)}")
    pd.DataFrame(missing_animal, columns=["pmid"]).to_csv(f"{output_dir}/missing_animal_filter.csv", index=False)

    # 4. All NER identified PMIDs
    ner_pmids = set(identified_studies['PMID'])
    missing_ner = target_pmids - ner_pmids
    print(f"Missing after general NER filter: {len(missing_ner)}")
    pd.DataFrame(missing_ner, columns=["pmid"]).to_csv(f"{output_dir}/missing_ner_filter.csv", index=False)

    print("✅ All missing lists saved independently.")
    
def main(fetch_dois=False, check_missing=True):
    #print(get_pmids_for_doi(doi="10.1016/j.intimp.2018.12.001"))
    if fetch_dois:
        process_dois(INPUT_CSV, OUTPUT_CSV)
    if check_missing:    
        hermes_df = pd.read_csv(OUTPUT_CSV)
        check_missing_pmids(hermes_df, output_dir="05_syst_reviews_validation/outputs/missing_ms_hermes_validation")
   
if __name__ == "__main__":
    main(fetch_dois=False, check_missing=True)