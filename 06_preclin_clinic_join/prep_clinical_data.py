import pandas as pd

clinical_df = pd.read_csv("06_preclin_clinic_join/data/clinical/aggregated_ner_annotations_basic_dict_mapped_19632.csv")
included_nctids = pd.read_csv("06_preclin_clinic_join/data/clinical/clinical_inlcuded_18609_nctids.csv")

clinical_df = clinical_df[clinical_df['nct_id'].isin(included_nctids['nct_id'])]

def combine_unique_entities_bert_aact(col1, col2):
    col1 = str(col1) if not pd.isna(col1) else ""
    col2 = str(col2) if not pd.isna(col2) else ""
    combined = set(col1.split('|')) | set(col2.split('|'))
    return '|'.join(sorted(combined))

# Merge canonical and BioLinkBERT-based annotations
clinical_df['linkbert_aact_mapped_conditions'] = clinical_df.apply(
    lambda row: combine_unique_entities_bert_aact(row['canonical_BioLinkBERT-base_conditions'], row['canonical_aact_conditions']),
    axis=1
)

clinical_df['linkbert_aact_mapped_drugs'] = clinical_df.apply(
    lambda row: combine_unique_entities_bert_aact(row['canonical_BioLinkBERT-base_interventions'], row['canonical_aact_interventions']),
    axis=1
)

# Keep relevant columns and explode for 1-to-1 condition-drug mapping
clinical_df = clinical_df[['nct_id', 'linkbert_aact_mapped_conditions', 'linkbert_aact_mapped_drugs', 'Disease Class']]
clinical_df.to_csv("06_preclin_clinic_join/data/clinical/clinical_combined_annotations.csv", index=False)