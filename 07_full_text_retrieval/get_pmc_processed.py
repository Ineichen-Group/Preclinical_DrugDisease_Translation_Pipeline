import glob


INPUT_DIR = "07_full_text_retrieval/pmc_fulltext"
disease_files = glob.glob(f"{INPUT_DIR}/*_fulltext")
print(f"Found {len(disease_files)} disease-specific files in {INPUT_DIR}")

pmids = set()  # To collect all PMIDs across files

for disease_file in disease_files:
    print(f"Processing {disease_file}...")
    pmid_files = glob.glob(f"{disease_file}/*.json")
    print(f"Found {len(pmid_files)} PMIDs in {disease_file}")
    for pmid_file in pmid_files:
        pmid = pmid_file.split("/")[-1].replace(".json", "")
        pmids.add(pmid)
        
print(f"Total unique PMIDs collected: {len(pmids)}")
output_file = "06_preclin_clinic_join/data/preclinical_for_full_text/already_processed_pmc_pmids.csv"
with open(output_file, "w") as f:
    f.write(f"PMID\n")
    for pmid in sorted(pmids):
        f.write(f"{pmid}\n")
        
        