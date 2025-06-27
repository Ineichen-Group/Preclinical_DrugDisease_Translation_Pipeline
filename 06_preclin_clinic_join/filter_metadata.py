import pandas as pd
import ast

# === FILE PATH PARAMETERS ===
MAPPED_STUDIES_PATH = "06_preclin_clinic_join/data/joined_data/condition_clinical_and_preclinical_12237.csv"

CLINICAL_METADATA_PATH = "06_preclin_clinic_join/data/clinical/clinical_nct_docs_metadata_20240313.csv"
PRECLINICAL_METADATA_PATH = "03_IE_ner/data/animal_studies_with_drug_disease/animal_studies_metadata_562352.csv"

CLINICAL_ANNOTATIONS_PATH = "04_normalization/data/mapped_all/mapped_clinical_data.csv"
PRECLINICAL_ANNOTATIONS_PATH = "04_normalization/data/mapped_all/mapped_preclinical_data.csv"

OUTPUT_CLINICAL_METADATA = "06_preclin_clinic_join/data/joined_data/clinical_metadata_mapped.csv"
OUTPUT_PRECLINICAL_METADATA = "06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv"

# === LOAD MAPPED STUDIES ===
mapped_studies = pd.read_csv(MAPPED_STUDIES_PATH)
mapped_studies['clinical_doc_ids'] = mapped_studies['clinical_doc_ids'].apply(ast.literal_eval)
mapped_studies['preclinical_doc_ids'] = mapped_studies['preclinical_doc_ids'].apply(ast.literal_eval)

# Flatten all clinical_doc_ids (NCTIDs) into a set
all_nctids = set(nctid for sublist in mapped_studies['clinical_doc_ids'] for nctid in sublist)

# Flatten all preclinical_doc_ids (PMIDs) into a set
all_pmids = set(pmid for sublist in mapped_studies['preclinical_doc_ids'] for pmid in sublist)

print(f"Unique clinical with mapping: {len(all_nctids)}, preclinical: {len(all_pmids)}")

# === LOAD AND FILTER METADATA ===
df_clinical_metadata = pd.read_csv(CLINICAL_METADATA_PATH)[[
    'nct_id','study_official_title','start_date','completion_date', 'phase','study_type','overall_status'
]]
df_clinical_metadata['start_year'] = pd.to_datetime(df_clinical_metadata['start_date']).dt.year
df_clinical_metadata['completion_year'] = pd.to_datetime(df_clinical_metadata['completion_date']).dt.year

df_preclinical_metadata = pd.read_csv(PRECLINICAL_METADATA_PATH)

df_clinical_annotations = pd.read_csv(CLINICAL_ANNOTATIONS_PATH)
df_preclinical_annotations = pd.read_csv(PRECLINICAL_ANNOTATIONS_PATH)

# filter only to the overalpping studies
df_clinical_metadata = df_clinical_metadata[df_clinical_metadata['nct_id'].isin(all_nctids)]
df_preclinical_metadata = df_preclinical_metadata[df_preclinical_metadata['PMID'].isin(all_pmids)]

df_clinical_metadata = pd.merge(df_clinical_metadata, df_clinical_annotations, how='left', on='nct_id')
df_preclinical_metadata = pd.merge(df_preclinical_metadata, df_preclinical_annotations, how='left', on='PMID')

# Drop duplicates
df_clinical_metadata = df_clinical_metadata.drop_duplicates()
df_preclinical_metadata = df_preclinical_metadata.drop_duplicates()

print(f"Clinical {df_clinical_metadata.shape}, preclinical {df_preclinical_metadata.shape}")

# === RENAME COLUMNS TO STANDARD FORMAT ===
df_clinical_metadata.rename(columns={
    'linkbert_aact_mapped_conditions': 'dict_disease',
    'linkbert_aact_mapped_drugs': 'dict_drug'
}, inplace=True)

df_clinical_metadata.rename(columns={
    'disease_term_mondo_norm': 'disease',
    'drug_term_umls_norm': 'drug'
}, inplace=True)

df_preclinical_metadata.rename(columns={
    'unique_conditions_linkbert_predictions': 'raw_disease',
    'unique_interventions_linkbert_predictions': 'raw_drug'
}, inplace=True)

df_preclinical_metadata.rename(columns={
    'disease_term_mondo_norm': 'disease',
    'drug_term_umls_norm': 'drug'
}, inplace=True)

# === SAVE OUTPUT ===
df_clinical_metadata.to_csv(OUTPUT_CLINICAL_METADATA, index=False)
df_preclinical_metadata.to_csv(OUTPUT_PRECLINICAL_METADATA, index=False)
