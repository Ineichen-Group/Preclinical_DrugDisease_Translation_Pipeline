import re
import pandas as pd
from typing import Set

def load_wrong_pmids(csv_paths: list) -> Set[str]:
    """
    Read multiple CSVs of “wrong” PMIDs (e.g. case reports, reviews) and return their union.
    """
    pmids = set()
    for p in csv_paths:
        df = pd.read_csv(p)
        if 'PMID' in df.columns:
            pmids |= set(df['PMID'].astype(str))
        elif 'pmid' in df.columns:
            pmids |= set(df['pmid'].astype(str))
    return pmids

csv_pmids_to_exclude = ["03_IE_ner/check_study_type/animal_studies_case_report_publications.csv",
                        "03_IE_ner/check_study_type/animal_studies_review_publications.csv",
                        "03_IE_ner/check_study_type/animal_studies_clinical_trial_publications.csv"]


def filter_and_save_ms_articles():
    save_dir = "03_IE_ner/data"

    # save MS related articles
    filtered_df_non_empty = pd.read_csv("03_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_sclerosis_10043.csv")
    pattern = r"\b(multiple sclerosis|ms)\b"

    filtered_df_sclerosis = filtered_df_non_empty[
        filtered_df_non_empty["unique_conditions_linkbert_predictions"]
        .str.contains(pattern, case=False, na=False, flags=re.IGNORECASE)
    ]
    filtered_df_sclerosis[["PMID", "unique_conditions_linkbert_predictions", "unique_interventions_linkbert_predictions"]].to_csv(f'{save_dir}/animal_studies_with_drug_disease/filtered_df_non_empty_MS_{len(filtered_df_sclerosis)}.csv',  index=False)
    filtered_df_sclerosis[["PMID"]].to_csv(
        f"{save_dir}/animal_studies_with_drug_disease/filtered_df_non_empty_MS_{len(filtered_df_sclerosis)}_PMIDs.csv",
        index=False
    )

def save_filtered_metadata():
    
    path_to_annotated_ner = "03_IE_ner/data/animal_studies_with_drug_disease"
    metadata_file = "02_animal_study_classification/data/animal_studies/full_pubmed_filtered_animal_6002827_metadata.csv"
    # save metadata of relevant articles
    metadata_full = pd.read_csv(metadata_file)
    metadata_full = metadata_full.drop_duplicates()
    metadata_full["PMID"] = metadata_full["PMID"].astype(str).str.strip()
    
    included_studies_main = pd.read_csv(f"{path_to_annotated_ner}/filtered_df_non_empty_595768.csv")
    included_studies_extra = pd.read_csv(f"{path_to_annotated_ner}/filtered_df_non_empty_4489.csv")
    included_studies = pd.concat([included_studies_main, included_studies_extra], ignore_index=True)
    
    wrong_pmids = load_wrong_pmids(csv_pmids_to_exclude)
    print(f"Loaded {len(wrong_pmids)} excluded PMIDs")
    included_studies["PMID"] = included_studies["PMID"].astype(str).str.strip()
    included_studies = included_studies[~included_studies["PMID"].isin(wrong_pmids)]
    print(f"preclinical {len(included_studies)} PMIDs remain after filtering")

    result = pd.merge(included_studies, metadata_full, how="left", on="PMID")
    print(f"studies metadata ", result.shape)
    result.to_csv(f"{path_to_annotated_ner}/animal_studies_metadata_{len(result)}.csv", index=False)

if __name__ == "__main__":
    save_filtered_metadata()