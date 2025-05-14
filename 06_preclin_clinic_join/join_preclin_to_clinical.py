import pandas as pd
import numpy as np
from viz_data import viz_joined_preclin_clinical
from viz_data import plot_top_entities_side_by_side

# ------------------------- #
#        LOAD DATA         #
# ------------------------- #

FILE_PRECLINICAL_LINKING = "04_normalization/data/mapped_all/mapped_preclinical_data.csv"
FILE_CLINICAL_LINKING = "04_normalization/data/mapped_all/mapped_clinical_data.csv"
FILE_CLINICAL_METADATA = "06_preclin_clinic_join/data/clinical/clinical_nct_docs_metadata_20240313.csv"

OUTPUT_DIR = "06_preclin_clinic_join/data/joined_data/"

conditions_col_to_use = "disease_term_mondo_norm" # #"linkbert_mapped_conditions" # 
drugs_col_to_use =  "drug_term_umls_norm" #"linkbert_mapped_drugs" #"drug_term_umls_norm"

conditions_col_to_use_clinical =  "disease_term_mondo_norm"#"linkbert_aact_mapped_conditions" #"disease_term_mondo_norm"
drugs_col_to_use_clinical =  "drug_term_umls_norm" #"linkbert_aact_mapped_drugs" # "drug_term_umls_norm"

# --- Load Preclinical Data ---
preclinical_df = pd.read_csv(FILE_PRECLINICAL_LINKING)
print(f"Shape of preclinical_df: {preclinical_df.shape}, {preclinical_df.PMID.nunique()} unique PMIDs")

# Split and explode conditions and drugs
preclinical_df[conditions_col_to_use] = preclinical_df[conditions_col_to_use].str.split("|")
preclinical_df = preclinical_df.explode(conditions_col_to_use, ignore_index=True)

preclinical_df[drugs_col_to_use] = preclinical_df[drugs_col_to_use].str.split("|")
preclinical_df = preclinical_df.explode(drugs_col_to_use, ignore_index=True)

preclinical_df = preclinical_df.drop_duplicates()

# Strip whitespace and convert to lowercase
preclinical_df[conditions_col_to_use] = preclinical_df[conditions_col_to_use].str.strip().str.lower()
preclinical_df[drugs_col_to_use] = preclinical_df[drugs_col_to_use].str.strip().str.lower()

# Create disease-drug key
preclinical_df['disease<>drug'] = (
    preclinical_df[conditions_col_to_use] + " <> " + preclinical_df[drugs_col_to_use]
)

plot_top_entities_side_by_side(preclinical_df, id_column='PMID', condition_column=conditions_col_to_use, drug_column=drugs_col_to_use, color_code='#56B4E9')

# --- Load Clinical Data ---
clinical_df = pd.read_csv(FILE_CLINICAL_LINKING)
print(f"Shape of clinical_df: {clinical_df.shape}, {clinical_df.nct_id.nunique()} unique NCTIDs")

clinical_df[conditions_col_to_use_clinical] = clinical_df[conditions_col_to_use_clinical].str.split("|")
clinical_df = clinical_df.explode(conditions_col_to_use_clinical, ignore_index=True)

clinical_df[drugs_col_to_use_clinical] = clinical_df[drugs_col_to_use_clinical].str.split("|")
clinical_df = clinical_df.explode(drugs_col_to_use_clinical, ignore_index=True)

# Strip whitespace and convert to lowercase
clinical_df[conditions_col_to_use_clinical] = clinical_df[conditions_col_to_use_clinical].str.strip().str.lower()
clinical_df[drugs_col_to_use_clinical] = clinical_df[drugs_col_to_use_clinical].str.strip().str.lower()

# Create disease-drug key
clinical_df['disease<>drug'] = (
    clinical_df[conditions_col_to_use_clinical] + " <> " + clinical_df[drugs_col_to_use_clinical]
)

plot_top_entities_side_by_side(clinical_df, id_column='nct_id', condition_column=conditions_col_to_use_clinical, drug_column=drugs_col_to_use_clinical, viz_name_suffix='clinical')

# Load and merge clinical metadata (phase + status)
metadata_df = pd.read_csv(FILE_CLINICAL_METADATA)[['nct_id', 'phase', 'overall_status']]
metadata_df = metadata_df.drop_duplicates()

clinical_df = clinical_df.merge(metadata_df, on='nct_id', how='left')

# ------------------------- #
#     AGGREGATE & MERGE     #
# ------------------------- #
def aggregate_and_merge(clinical_df, preclinical_df, 
                        clinical_key_col, clinical_doc_id_col, 
                        preclinical_key_col, preclinical_doc_id_col,
                        preclinical_disease_col, preclinical_drug_col):
    """
    Aggregates and merges clinical and preclinical data on a shared normalized key.

    Parameters:
    ----------
    clinical_df : pd.DataFrame
        DataFrame containing clinical study data, including a merge key and document-level info.

    preclinical_df : pd.DataFrame
        DataFrame containing preclinical (animal study) data with a similar merge key and document info.

    clinical_key_col : str
        Column name in the clinical dataframe that holds the disease<>drug key used for merging.

    clinical_doc_id_col : str
        Column name in the clinical dataframe representing document IDs (e.g., NCT IDs).

    preclinical_key_col : str
        Column name in the preclinical dataframe that holds the disease<>drug key used for merging.

    preclinical_doc_id_col : str
        Column name in the preclinical dataframe representing document IDs (e.g., PMIDs).

    preclinical_disease_col : str
        Column name in the preclinical dataframe containing the mapped disease name(s).

    preclinical_drug_col : str
        Column name in the preclinical dataframe containing the mapped drug name(s).

    Returns:
    -------
    pd.DataFrame
        A merged DataFrame containing aggregated clinical and preclinical data by normalized key.
        Includes counts of associated documents and selected metadata like disease, drug, phase, and status.
    """

    # === Aggregate clinical data by key ===
    clinical_agg = clinical_df.groupby(clinical_key_col).agg({
        clinical_doc_id_col: list,
        'phase': list,
        'overall_status': list
    }).reset_index()

    # Rename columns for clarity and consistency
    clinical_agg.rename(columns={
        clinical_key_col: 'normalized_key',
        clinical_doc_id_col: 'clinical_doc_ids'
    }, inplace=True)

    # Add count of clinical documents per key
    clinical_agg['clinical_count'] = clinical_agg['clinical_doc_ids'].apply(len)

    # === Aggregate preclinical data by key ===
    preclinical_agg = preclinical_df.groupby(preclinical_key_col).agg({
        preclinical_doc_id_col: list,
        preclinical_disease_col: 'first',
        preclinical_drug_col: 'first'
    }).reset_index()

    # Rename columns for consistency
    preclinical_agg.rename(columns={
        preclinical_key_col: 'normalized_key',
        preclinical_doc_id_col: 'preclinical_doc_ids',
        preclinical_disease_col: 'disease',
        preclinical_drug_col: 'drug'
    }, inplace=True)

    # Add count of preclinical documents per key
    preclinical_agg['preclinical_count'] = preclinical_agg['preclinical_doc_ids'].apply(len)

    # === Merge clinical and preclinical aggregates on the normalized key ===
    merged_df = pd.merge(clinical_agg, preclinical_agg, on='normalized_key', how='outer')

    return merged_df

def clean_and_sort_study_data(df):
    """
    Cleans and sorts a merged clinical/preclinical DataFrame by removing invalid rows
    and ordering by study counts.

    Specifically:
    - Sorts the DataFrame in descending order of 'clinical_count' and 'preclinical_count'.
    - Removes rows where either 'clinical_count' or 'preclinical_count' is NaN.
    - Removes rows where the 'drug' name is missing or has 3 or fewer characters.
    - Prints the shape before and after filtering for transparency.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing 'clinical_count', 'preclinical_count', and 'drug'.

    Returns:
    -------
    pd.DataFrame
        The cleaned and sorted DataFrame.
    """
    print(f'Input shape of merged for cleaning: {df.shape}')

    # Sort by descending clinical and preclinical study counts
    sorted_df = df.sort_values(
        by=['clinical_count', 'preclinical_count'],
        ascending=[False, False]
    )

    # Remove rows with missing study counts
    filtered_df = sorted_df.dropna(subset=['clinical_count', 'preclinical_count'])

    # Remove rows with invalid or too-short drug names
    filtered_df = filtered_df[
        filtered_df['drug'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 3)
    ]

    print(f'Shape after filtering empty counts and short drugs: {filtered_df.shape}')
    return filtered_df


# Print unique disease-drug pairs from clinical and preclinical
# Print unique disease-drug pairs from clinical and preclinical
# Preclinical Stats
print("----- Preclinical Stats: -----")
print("Unique pmid in preclinical data:")
print(preclinical_df['PMID'].nunique())
print("Unique disease-drug pairs in preclinical data:")
preclinical_unique_pairs = preclinical_df['disease<>drug'].value_counts().reset_index()
preclinical_unique_pairs.columns = ['disease<>drug', 'count']
print(preclinical_unique_pairs.shape)

print("Unique disease-drug pairs with count larger than 1 in preclinical data:")
preclinical_unique_pairs_count = preclinical_unique_pairs[preclinical_unique_pairs['count'] > 1]
print(preclinical_unique_pairs_count.shape)

# Save unique pairs to CSV
preclinical_unique_pairs.to_csv(f'{OUTPUT_DIR}preclinical_unique_pairs.csv', index=False)

# Clinical Stats
print("----- Clinical Stats: -----")
print("Unique nct_id in clinical data:")
print(clinical_df['nct_id'].nunique())
print("Unique disease-drug pairs in clinical data:")
clinical_unique_pairs = clinical_df['disease<>drug'].value_counts().reset_index()
clinical_unique_pairs.columns = ['disease<>drug', 'count']
print(clinical_unique_pairs.shape)

print("Unique disease-drug pairs with count larger than 1 in clinical data:")
clinical_unique_pairs_count = clinical_unique_pairs[clinical_unique_pairs['count'] > 1]
print(clinical_unique_pairs_count.shape)

# Save unique pairs to CSV
clinical_unique_pairs.to_csv('{OUTPUT_DIR}clinical_unique_pairs.csv', index=False)

# Apply aggregation + filtering
merged_df = aggregate_and_merge(
    clinical_df=clinical_df,
    preclinical_df=preclinical_df,
    clinical_key_col='disease<>drug',
    clinical_doc_id_col='nct_id',
    preclinical_key_col='disease<>drug',
    preclinical_doc_id_col='PMID',
    preclinical_disease_col=conditions_col_to_use, 
    preclinical_drug_col=drugs_col_to_use
)

filtered_df = clean_and_sort_study_data(merged_df)

print("----- Joined Stats: -----")
print(f"Filtered shape: {filtered_df.shape}")

# Flatten and count unique clinical and preclinical IDs
clinical_ids = filtered_df['clinical_doc_ids'].dropna().explode()
preclinical_ids = filtered_df['preclinical_doc_ids'].dropna().explode()

print(f"Unique clinical doc IDs after merge: {clinical_ids.nunique()}")
print(f"Unique preclinical doc IDs after merge: {preclinical_ids.nunique()}")

# ------------------------- #
#   ADD INSIGHTS / EXPORT   #
# ------------------------- #

phase_order = {
    'Early Phase 1': 0,
    'Phase 1': 1,
    'Phase 1/Phase 2': 1.5,
    'Phase 2': 2,
    'Phase 2/Phase 3': 2.5,
    'Phase 3': 3,
    'Phase 4': 4,
    'Not Applicable': -1  # Lowest value to ignore in max comparison
}

# Get max phase for each row
def get_max_phase(phases):
    max_val = -1
    max_phase = 'Not Applicable'
    for phase in phases:
        val = phase_order.get(phase, -1)
        if val > max_val:
            max_val = val
            max_phase = phase
    return max_phase

filtered_df['max_phase'] = filtered_df['phase'].apply(get_max_phase)

# Identify studies with Phase 3 or 4 trials
filtered_df['at_least_one_phase3'] = filtered_df['phase'].apply(
    lambda phases: any("Phase 3" in str(p) for p in phases)
)

filtered_df['at_least_one_phase4'] = filtered_df['phase'].apply(
    lambda phases: any("Phase 4" in str(p) for p in phases)
)

# Summary stats
total = len(filtered_df)
pct_phase3 = (filtered_df['at_least_one_phase3'].sum() / total) * 100
pct_phase4 = (filtered_df['at_least_one_phase4'].sum() / total) * 100

print(f"Percentage with Phase 3: {pct_phase3:.2f}%")
print(f"Percentage with Phase 4: {pct_phase4:.2f}%")

# Export for manual review
output_path = f"{OUTPUT_DIR}condition_clinical_and_preclinical_{total}.csv"
filtered_df.to_csv(output_path, index=False)

# ------------------------- #
#         VISUALIZE         #
# ------------------------- #

viz_joined_preclin_clinical(
    filtered_df,
    "normalized_key",
    translation_column='at_least_one_phase4',
    top_n=20,
    fig_name_suffix='_disease_drug'
)
