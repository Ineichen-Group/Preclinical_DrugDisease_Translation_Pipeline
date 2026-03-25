
import pandas as pd

def save_filtered_metadata():
    
    path_to_annotated_ner = "03_IE_ner/data/animal_studies_with_drug_disease"
    metadata_file = "02_animal_study_classification/data/animal_studies/full_pubmed_filtered_animal_6002827_metadata.csv"
    # save metadata of relevant articles
    metadata_full = pd.read_csv(metadata_file)
    metadata_full = metadata_full.drop_duplicates()
    metadata_full["PMID"] = metadata_full["PMID"].astype(str).str.strip()
    
    included_studies_main = pd.read_csv(f"{path_to_annotated_ner}/filtered_df_non_empty_595768.csv")
    included_studies_extra = pd.read_csv(f"{path_to_annotated_ner}/filtered_df_non_empty_2879.csv")
    included_studies = pd.concat([included_studies_main, included_studies_extra], ignore_index=True)
    included_studies["PMID"] = included_studies["PMID"].astype(str).str.strip()
    
    result = pd.merge(included_studies, metadata_full, how="left", on="PMID")
    print(f"studies metadata ", result.shape)
    result.to_csv(f"{path_to_annotated_ner}/animal_studies_metadata_{len(result)}.csv", index=False)
    print(f"Saved filtered metadata with {len(result)} entries to {path_to_annotated_ner}/animal_studies_metadata_{len(result)}.csv")

if __name__ == "__main__":
    save_filtered_metadata()