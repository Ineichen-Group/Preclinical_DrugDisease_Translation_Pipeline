import pandas as pd
import ast
import argparse
import os

# === CONFIGURATION ===
# Maps label group names to a tuple:
# (path to CSV, {source column name: target column name}, parse_list flag)
def build_annotation_files(base_dir):
    return {
        'animal_sex': (
            os.path.join(base_dir, "regex/server/sex_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'animal_sex'},
            False
        ),
        'animal_species': (
            os.path.join(base_dir, "regex/server/species_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'animal_species'},
            False
        ),
        'animal_age': (
            os.path.join(base_dir, "age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions_mapped_20250722.csv"),
            {
                'age_classification': 'animal_age_class',
                'prediction_normalized_age': 'animal_age'
            },
            False
        ),
        'rigor_blinding': (
            os.path.join(base_dir, "regex/server/blinding_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'rigor_blinding'},
            False
        ),
        'rigor_randomization': (
            os.path.join(base_dir, "regex/server/randomization_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'rigor_randomization'},
            False
        ),
        'rigor_welfare': (
            os.path.join(base_dir, "regex/server/welfare_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'rigor_welfare'},
            False
        ),
        'assay_type': (
            os.path.join(base_dir, "regex/server/assay_doc_level_predictions.csv"),
            {'prediction_encoded_label': 'assay_type'},
            False
        ),
        'animal_strain': (
            os.path.join(base_dir, "strain/strain_predictions_normalized.csv"),
            {'prediction_encoded_label': 'animal_strain'},
            False
        ),
        'animal_number': (
            os.path.join(base_dir, "animals_nr/animals_nr_predictions_numeric.csv"),
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
    parser.add_argument(
        "--base_annotation_dir",
        default="08_IE_full_text/model_predictions",
        help="Base directory where annotation prediction CSVs are located."
    )
    parser.add_argument(
        "--metadata",
        default="06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv",
        help="Path to preclinical metadata CSV."
    )
    parser.add_argument(
        "--fulltext_pmids",
        default="07_full_text_retrieval/materials_methods/combined/combined_methods.jsonl",
        help="JSONL file with PMIDs and full text content (used to check full text availability)."
    )
    parser.add_argument(
        "--output",
        default="06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped_annotated.csv",
        help="Path to save final annotated output CSV."
    )
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata)
    print(f"Read metadata: {args.metadata} with {metadata_df.shape[0]} rows")

    fulltext_df = pd.read_json(args.fulltext_pmids, lines=True)
    fulltext_pmids = fulltext_df["PMID"].unique()
    print(f"Read full-text PMIDs from JSONL: {args.fulltext_pmids} with {len(fulltext_pmids)} unique PMIDs")

    merged = metadata_df.copy()
    ANNOTATION_FILES = build_annotation_files(args.base_annotation_dir)
    # Apply annotations
    for label_group, (file_path, col_rename_map, parse_list) in ANNOTATION_FILES.items():
        print(f"\nProcessing annotation group: {label_group}")
        annot_df = load_annotation(file_path, col_rename_map, parse_list=parse_list)
        merged = add_annotation(merged, annot_df, list(col_rename_map.values()), fulltext_pmids=fulltext_pmids)

    # Summary
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
