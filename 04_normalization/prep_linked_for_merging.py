import pandas as pd
import numpy as np


def load_and_merge_preclinical_data():    
    """
    Load preclinical data and linked ontologies, merge them on 'PMID',
    and save the merged DataFrame.
    """
    # --- Load Preclinical Data ---
    preclinical_df_main = pd.read_csv("04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_595768.csv")
    print(f"Shape of preclinical_df_main: {preclinical_df_main.shape}")
    preclinical_df_extra = pd.read_csv("04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_4489.csv")
    print(f"Shape of preclinical_df_extra: {preclinical_df_extra.shape}")
    preclinical_df = pd.concat([preclinical_df_main, preclinical_df_extra], ignore_index=True)

    print(f"Shape of preclinical_df before dropping duplicates: {preclinical_df.shape}")
    preclinical_df = preclinical_df.drop_duplicates(subset='PMID', keep='first')
    print(f"Shape of preclinical_df after dropping duplicates: {preclinical_df.shape}")

    preclinical_linked_ontologies_main = pd.read_csv("04_normalization/data/mapped_to_embeddings_ontologies/drug_disease_mapped_preclinical.csv")
    preclinical_linked_ontologies_extra = pd.read_csv(
        "04_normalization/data/mapped_to_embeddings_ontologies/drug_disease_mapped_preclinical_extra_studies.csv"
    )
    # Rename columns for consistency
    preclinical_linked_ontologies_extra = preclinical_linked_ontologies_extra.rename(columns={
        'umls_term_norm': 'drug_term_umls_norm',
        'umls_termid': 'drug_umls_termid',
        'mondo_term_norm': 'disease_term_mondo_norm',
        'mondo_termid': 'disease_mondo_termid'
    })

    # Select and reorder relevant columns
    preclinical_linked_ontologies_extra = preclinical_linked_ontologies_extra[[
        "PMID",
        "disease_term_mondo_norm",
        "disease_mondo_termid",
        "drug_term_umls_norm",
        "drug_umls_termid"
    ]]
    preclinical_linked_ontologies = pd.concat([preclinical_linked_ontologies_main, preclinical_linked_ontologies_extra], ignore_index=True)

    # Print the shapes of the loaded DataFrames
    print(f"Shape of preclinical_df: {preclinical_df.shape}, {preclinical_df.PMID.nunique()} unique PMIDs")
    print(f"Shape of preclinical_linked_ontologies: {preclinical_linked_ontologies.shape}")

    # Perform a left join on 'PMID'
    merged_df = preclinical_df.merge(preclinical_linked_ontologies, on="PMID", how="left")

    # Print the shape of the merged DataFrame
    print(f"Shape of merged_df: {merged_df.shape}")

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv("04_normalization/data/mapped_all/mapped_preclinical_data.csv", index=False)


def load_and_merge_clinical_data():
    """    Load clinical data and linked ontologies, merge them on 'nct_id',
    and save the merged DataFrame.
    """
    # --- Load Clinical Data ---
    clinical_df = pd.read_csv("06_preclin_clinic_join/data/clinical/clinical_combined_annotations.csv")
    clinical_linked_ontologies = pd.read_csv("04_normalization/data/mapped_to_embeddings_ontologies/drug_disease_mapped_clinical.csv")

    # Print the shapes of the loaded DataFrames
    print(f"Shape of clinical_df: {clinical_df.shape}")
    print(f"Shape of clinical_linked_ontologies: {clinical_linked_ontologies.shape}")

    # Perform a left join on 'PMID'
    merged_df = clinical_df.merge(clinical_linked_ontologies, on="nct_id", how="left")

    # Print the shape of the merged DataFrame
    print(f"Shape of merged_df: {merged_df.shape}")

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv("04_normalization/data/mapped_all/mapped_clinical_data.csv", index=False)

if __name__ == "__main__":
    #load_and_merge_clinical_data()
    load_and_merge_preclinical_data()
    print("Data merging completed successfully.")