import pandas as pd
import ast

def load_annotation(file_path, label_name, parse_list=False):
    """Load annotation file and rename the prediction column."""
    df = pd.read_csv(file_path)[['PMID', 'prediction_encoded_label']]
    df = df.rename(columns={'prediction_encoded_label': label_name})
    if parse_list:
        df[label_name] = df[label_name].apply(
            lambda x: ', '.join(ast.literal_eval(x)) if pd.notna(x) else x
        )
    return df

def add_annotation(main_df, annot_df, label_name):
    """Merge annotation and fill missing values."""
    merged_df = main_df.merge(annot_df, on='PMID', how='left')
    merged_df[label_name] = merged_df[label_name].fillna('unlabeled')
    return merged_df

# --- Main process ---

# Load preclinical metadata
preclin_metadata_file = "06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv"
preclin_metadata = pd.read_csv(preclin_metadata_file)
print(f'preclin_metadata: {preclin_metadata.shape}')

# Annotation files and settings
annotations = {
    'animal_sex':    ("08_IE_full_text/model_predictions/sex/sex_predictions_MS.csv", False),
    'animal_species':("08_IE_full_text/model_predictions/species/species_predictions_MS.csv", True),
    'animal_strain': ("08_IE_full_text/model_predictions/strain/strain_predictions_MS.csv", False)
}

# Process and merge all annotations
merged = preclin_metadata.copy()
for label, (file_path, parse_list) in annotations.items():
    annot_df = load_annotation(file_path, label, parse_list=parse_list)
    merged = add_annotation(merged, annot_df, label)

# Count 'unlabeled' values
for label in annotations.keys():
    count = (merged[label] == 'unlabeled').sum()
    print(f"Unlabeled {label} count: {count}")

# Save
print(f'merged with annotations {merged.shape}')
merged.to_csv("06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped_annotated.csv")




