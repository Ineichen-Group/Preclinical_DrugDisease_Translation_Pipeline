import pandas as pd
import os
from datetime import datetime

# Input files
target_file = "03_IE_ner/data/animal_studies_with_drug_disease/animal_studies_metadata_after_stype_filter_554307_PMIDs.csv" #"03_IE_ner/data/animal_studies_with_drug_disease/disease_filtered/all_filtered_disease_pmids_47059.csv"
already_processed_files = [
    "06_preclin_clinic_join/data/preclinical_for_full_text/pmids_to_fetch_20251507.csv",
    "06_preclin_clinic_join/data/preclinical_for_full_text/already_processed_pmc_pmids.csv",
    "06_preclin_clinic_join/data/preclinical_for_full_text/already_processed_cadmus_pmids.csv"
    "07_full_text_retrieval/pmids_to_retrieve/pmids_remaining_disease_selection_29529_20250725.csv"
]

# Output path
save_filtered_path = "07_full_text_retrieval/pmids_to_retrieve"
os.makedirs(save_filtered_path, exist_ok=True)

# Load target PMIDs
target_pmids = pd.read_csv(target_file)
target_pmids["PMID"] = target_pmids["PMID"].astype(str)

# Combine already processed PMIDs
processed_pmids = set()
for file_path in already_processed_files:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if "PMID" in df.columns:
            processed_pmids.update(df["PMID"].dropna().astype(str))
        else:
            processed_pmids.update(df.iloc[:, 0].dropna().astype(str))  # Fallback if no column name
    else:
        print(f"Warning: File not found: {file_path}")

# Filter remaining PMIDs
remaining_pmids = target_pmids[~target_pmids["PMID"].isin(processed_pmids)]

# Save result
timestamp = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD

output_file = os.path.join(
    save_filtered_path,
    f"pmids_remaining_all_{len(remaining_pmids)}_{timestamp}.csv")
remaining_pmids.to_csv(output_file, index=False)

print(f"Saved {len(remaining_pmids)} remaining PMIDs to: {output_file}")
