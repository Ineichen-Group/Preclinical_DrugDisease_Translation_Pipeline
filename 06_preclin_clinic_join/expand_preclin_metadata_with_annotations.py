import pandas as pd
import ast
import argparse
import os

# === CONFIGURATION ===
# Maps label group names to a tuple:
# (path to CSV, {source column name: target column name}, parse_list flag)
ANNOTATION_FILES = {
    'animal_sex': (
        "08_IE_full_text/model_predictions/regex/sex_doc_level_predictions.csv",
        {'prediction_encoded_label': 'animal_sex'},
        False
    ),
    'animal_species': (
        "08_IE_full_text/model_predictions/regex/species_doc_level_predictions.csv",
        {'prediction_encoded_label': 'animal_species'},
        False
    ),
    'animal_age': (
        "08_IE_full_text/model_predictions/age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions_mapped.csv",
        {
            'age_classification': 'animal_age_class',
            'prediction_normalized_age': 'animal_age'
        },
        False
    ),
    'rigor_blinding': (
        "08_IE_full_text/model_predictions/regex/blinding_doc_level_predictions.csv",
        {'prediction_encoded_label': 'rigor_blinding'},
        False
    ),
    'rigor_randomization': (
        "08_IE_full_text/model_predictions/regex/randomization_doc_level_predictions.csv",
        {'prediction_encoded_label': 'rigor_randomization'},
        False
    ),
    'rigor_welfare': (
        "08_IE_full_text/model_predictions/regex/welfare_doc_level_predictions.csv",
        {'prediction_encoded_label': 'rigor_welfare'},
        False
    ),
    'assay_type': (
        "08_IE_full_text/model_predictions/regex/assay_doc_level_predictions.csv",
        {'prediction_encoded_label': 'assay_type'},
        False
    ),
    'animal_strain': (
        "08_IE_full_text/model_predictions/strain/strain_predictions_norm.csv",
        {'prediction_encoded_label': 'animal_strain'},
        False
    ),
    'animal_number': (
        "08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions_numeric.csv",
        {'prediction_encoded_label': 'animal_number'},
        False
    )
}

def load_annotation(file_path, col_rename_map, parse_list=False):
    """
    Load a single annotation CSV file and select/rename columns as needed.

    Args:
        file_path (str): Path to the CSV file containing predictions.
        col_rename_map (dict): Mapping from column names in the file to desired output names.
        parse_list (bool): Whether to parse stringified lists into actual strings (e.g. ['a', 'b'] → 'a, b').

    Returns:
        pd.DataFrame: DataFrame with 'PMID' and renamed annotation columns.
    """
    print(f'Loading annotation from: {file_path}')
    df = pd.read_csv(file_path)

    # Select only required columns
    required_cols = ['PMID'] + list(col_rename_map.keys())
    if missing := (set(required_cols) - set(df.columns)):
        raise ValueError(f"Missing columns in {file_path}: {missing}")
    
    df = df[required_cols].rename(columns=col_rename_map)

    # Optionally convert stringified lists to comma-separated strings
    if parse_list:
        for col in col_rename_map.values():
            df[col] = df[col].apply(
                lambda x: ', '.join(ast.literal_eval(x)) if pd.notna(x) else x
            )

    return df

def add_annotation(main_df, annot_df, new_cols, fulltext_pmids=None):
    """
    Merge one annotation DataFrame into the main metadata and fill missing labels.

    Args:
        main_df (pd.DataFrame): Preclinical metadata (with 'PMID').
        annot_df (pd.DataFrame): One annotation result table to merge.
        new_cols (list): Names of the new columns added.
        fulltext_pmids (iterable): Set of PMIDs with full-text available.

    Returns:
        pd.DataFrame: Updated metadata with merged annotation columns.
    """
    merged = main_df.merge(annot_df, on='PMID', how='left')

    for col in new_cols:
        if fulltext_pmids is not None:
            is_missing = merged[col].isna()
            has_text = merged['PMID'].isin(fulltext_pmids)

            # Assign fallback labels for missing values
            merged.loc[is_missing & has_text, col] = 'unlabeled'
            merged.loc[is_missing & ~has_text, col] = 'no full text'
        else:
            merged[col] = merged[col].fillna('unlabeled')

    return merged

def main():
    parser = argparse.ArgumentParser(description="Join preclinical metadata with prediction annotations.")
    parser.add_argument("--metadata", default="06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv", help="Path to preclinical metadata.")
    parser.add_argument("--fulltext_pmids", default="07_full_text_retrieval/materials_methods/combined/combined_methods.csv", help="CSV with PMIDs that have full text.")
    parser.add_argument("--output", default="06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped_annotated.csv", help="Path to save final output.")
    args = parser.parse_args()

    print(f"Reading metadata: {args.metadata}")
    metadata_df = pd.read_csv(args.metadata)
    fulltext_pmids = pd.read_csv(args.fulltext_pmids)["PMID"].unique()
    merged = metadata_df.copy()

    for label_group, (file_path, col_rename_map, parse_list) in ANNOTATION_FILES.items():
        print(f"\nProcessing annotation group: {label_group}")
        annot_df = load_annotation(file_path, col_rename_map, parse_list=parse_list)
        merged = add_annotation(merged, annot_df, list(col_rename_map.values()), fulltext_pmids=fulltext_pmids)

    # Summary of missing values
    print("\nAnnotation summary:")
    for _, (_, col_map, _) in ANNOTATION_FILES.items():
        for label in col_map.values():
            unlab = (merged[label] == 'unlabeled').sum()
            noft = (merged[label] == 'no full text').sum()
            print(f"  {label}: {len(merged) - unlab - noft} annotated | {unlab} unlabeled | {noft} no full text")

    print(f"\nSaving annotated dataset to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
