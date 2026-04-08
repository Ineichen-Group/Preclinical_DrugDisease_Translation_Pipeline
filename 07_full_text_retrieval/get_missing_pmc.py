import pandas as pd
import os

target_to_fetch_file = "07_full_text_retrieval/pmids_to_retrieve/pmids_remaining_disease_selection_29529_20250725.csv"
target_pmids = pd.read_csv(target_to_fetch_file, header=None, names=["PMID"])
print(f"Loaded {len(target_pmids)} PMIDs to fetch.")

success_files_folder = "07_full_text_retrieval/pmc_fulltext/preclin_disease_filtered_pmids_fulltext"
log_file_to_save = "07_full_text_retrieval/materials_methods/logs/pmc/no_methods_docs_pmc_selected_pmids.txt"

# read all JSON files in the success folder and extract PMID names from the filenames
# e.g. name methods_subtitles_8976172.json
success_files = os.listdir(success_files_folder)
success_pmids = set()
for file_name in success_files:
    if file_name.endswith(".json"):
        pmid = file_name.split("_")[-1].replace(".json", "")
        success_pmids.add(pmid)
        
# find PMIDs that are in target_pmids but not in success_pmids
missing_pmids = target_pmids[~target_pmids["PMID"].isin(success_pmids)]
print(f"Found {len(missing_pmids)} PMIDs that are missing from the success folder, percentage of total: {len(missing_pmids) / len(target_pmids) * 100:.2f}%")    

# save the missing PMIDs to a log file
with open(log_file_to_save, "w") as log_file:
    for pmid in missing_pmids["PMID"]:
        if pmid == "PMID":
            continue
        # write each PMID on a new line
        log_file.write(f"{pmid}\n")
print(f"Missing PMIDs saved to {log_file_to_save}")
