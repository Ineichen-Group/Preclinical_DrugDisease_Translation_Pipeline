import pandas as pd

pmids_to_fetch = pd.read_csv("06_preclin_clinic_join/data/preclinical_for_full_text/all/correct_study_type_pmids.csv")

already_processed_pmc = pd.read_csv("06_preclin_clinic_join/data/preclinical_for_full_text/already_processed_pmc_pmids.csv")
already_processed_pmc = set(already_processed_pmc['PMID'])
pmids_to_fetch = set(pmids_to_fetch['PMID'])
pmids_to_fetch = pmids_to_fetch - already_processed_pmc
print(f"PMIDs to fetch after removing PMC: {len(pmids_to_fetch)}")

already_processed_cadmus = pd.read_csv("06_preclin_clinic_join/data/preclinical_for_full_text/already_processed_cadmus_pmids.csv")
already_processed_cadmus = set(already_processed_cadmus['PMID'])
pmids_to_fetch = pmids_to_fetch - already_processed_cadmus
print(f"PMIDs to fetch after removing already processed Cadmus: {len(pmids_to_fetch)}")

output_file = "06_preclin_clinic_join/data/preclinical_for_full_text/pmids_to_fetch_20251507.csv"
with open(output_file, "w") as f:
    f.write(f"PMID\n")
    for pmid in sorted(pmids_to_fetch):
        f.write(f"{pmid}\n")