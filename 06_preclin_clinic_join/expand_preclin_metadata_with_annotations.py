import pandas as pd
import ast

# === CONFIGURATION ===

# Annotation files and their settings (label name: (file path, parse_list))
ANNOTATION_FILES = {
    'animal_sex':    ("08_IE_full_text/model_predictions/regex/sex_doc_level_predictions.csv", False),
    'animal_species':("08_IE_full_text/model_predictions/regex/species_doc_level_predictions.csv", False),
    'animal_age':    ("08_IE_full_text/model_predictions/age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions_mapped.csv", False),
    
    'rigor_blinding':    ("08_IE_full_text/model_predictions/regex/blinding_doc_level_predictions.csv", False),
    'rigor_randomization':    ("08_IE_full_text/model_predictions/regex/randomization_doc_level_predictions.csv", False),
    'rigor_welfare':    ("08_IE_full_text/model_predictions/regex/welfare_doc_level_predictions.csv", False),
    'assay_type':    ("08_IE_full_text/model_predictions/regex/assay_doc_level_predictions.csv", False),
    
    'animal_strain': ("08_IE_full_text/model_predictions/strain/strain_predictions_norm.csv", False),
    'animal_number': ("08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions_numeric.csv", False)
}

# === FUNCTIONS ===

def load_annotation(file_path, label_name, parse_list=False):
    """Load annotation file and rename the prediction column."""
    df = pd.read_csv(file_path)[['PMID', 'prediction_encoded_label']]
    df = df.rename(columns={'prediction_encoded_label': label_name})
    if parse_list:
        df[label_name] = df[label_name].apply(
            lambda x: ', '.join(ast.literal_eval(x)) if pd.notna(x) else x
        )
    return df

def add_annotation(main_df, annot_df, label_name, fulltext_pmids=None):
    """Merge annotation and label missing values based on full text availability."""
    merged_df = main_df.merge(annot_df, on='PMID', how='left')

    if fulltext_pmids is not None:
        is_missing = merged_df[label_name].isna()
        has_full_text = merged_df['PMID'].isin(fulltext_pmids)

        # Assign fallback labels
        merged_df.loc[is_missing & has_full_text, label_name] = 'unlabeled'
        merged_df.loc[is_missing & ~has_full_text, label_name] = 'no full text'
    else:
        merged_df[label_name] = merged_df[label_name].fillna('unlabeled')

    return merged_df

# === MAIN EXECUTION ===

# File paths
PRECLIN_METADATA_FILE = "06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv"
FULLTEXT_PMID_FILE = "07_full_text_retrieval/materials_methods/combined/combined_methods.csv"
OUTPUT_FILE = "06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped_annotated.csv"

# Load preclinical metadata
preclin_metadata = pd.read_csv(PRECLIN_METADATA_FILE)
print(f'preclin_metadata: {preclin_metadata.shape}')

# Load full text PMIDs
fulltext_pmids = pd.read_csv(FULLTEXT_PMID_FILE)["PMID"].unique()

# Merge all annotations
merged = preclin_metadata.copy()
for label, (file_path, parse_list) in ANNOTATION_FILES.items():
    print(f'Loading annotation for {label} from {file_path}')
    annot_df = load_annotation(file_path, label, parse_list=parse_list)
    merged = add_annotation(merged, annot_df, label, fulltext_pmids=fulltext_pmids)

# Count and report missing labels
for label in ANNOTATION_FILES.keys():
    unlabeled_count = (merged[label] == 'unlabeled').sum()
    no_fulltext_count = (merged[label] == 'no full text').sum()
    print(f"Unlabeled {label} count: {unlabeled_count}")
    print(f"No full text {label} count: {no_fulltext_count}")
    print(f"Total {label} count: {merged[label].notna().sum()}")

# Save final annotated metadata
print(f'Merged with annotations: {merged.shape}; saving to {OUTPUT_FILE}')
merged.to_csv(OUTPUT_FILE, index=False)
