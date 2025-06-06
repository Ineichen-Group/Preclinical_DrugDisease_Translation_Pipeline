import pandas as pd
import os
from pathlib import Path

###################### TODO: this should happen earlier in the pipeline
# List your input files
files = [
    '03_IE_ner/check_study_type/animal_studies_case_report_publications.csv',
    '03_IE_ner/check_study_type/animal_studies_clinical_trial_publications.csv',
    '03_IE_ner/check_study_type/animal_studies_review_publications.csv'
]

pmid_set_to_exclude = set()

for file in files:
    df = pd.read_csv(file)
    pmid_set_to_exclude.update(df['PMID'].dropna().astype(str).tolist())

print(f"Total PMIDs collected to exclude: {len(pmid_set_to_exclude)}")

######################

# Path to 'materials_methods'
base_dir = Path('07_full_text_retrieval/materials_methods')

# List to collect all data
all_data = []
count_skipped = 0
# Loop through all folders inside 'materials_methods'
for subfolder in base_dir.iterdir():
    ms_methods_dir = subfolder / 'MS_methods'
    if ms_methods_dir.exists():
        for csv_file in ms_methods_dir.glob('*.csv'):
            df = pd.read_csv(csv_file)
            # Check if 'pmid' exists, otherwise use 'doc_id'
            if 'pmid' not in df.columns and 'doc_id' in df.columns:
                df = df.rename(columns={'doc_id': 'pmid'})
                
            df['pmid'] = df['pmid'].astype(str)
            if df['pmid'].iloc[0] in pmid_set_to_exclude:
                count_skipped += 1
                continue
                
            # Step 1: Merge subtitle and paragraph into one column
            df['subtitle_paragraph'] = df['subtitle'].fillna('') + ' ' + df['paragraph'].fillna('')

            # Step 2: Group by pmid and join all subtitle_paragraphs into one big text
            df_merged = df.groupby('pmid')['subtitle_paragraph'].apply(lambda x: ' '.join(x)).reset_index()

            # Optional: rename for clarity
            df_merged = df_merged.rename(columns={'subtitle_paragraph': 'Text'})
            df_merged['source'] = subfolder.name
            all_data.append(df_merged)

# Combine all DataFrames
final_df = pd.concat(all_data, ignore_index=True)

# Rename columns
final_df.columns = ['PMID', 'Text', 'Source']
print(f'Count skipped rows due to wrong publication type: {count_skipped}')
print(f"Number of unique articles before dedup: {final_df['PMID'].nunique()}")

# Remove duplicate PMIDs, keeping the first occurrence
final_df = final_df.drop_duplicates(subset='PMID')

# Print the number of unique articles
print(f"Number of unique articles: {final_df['PMID'].nunique()}")

# Save to CSV
output_file = base_dir / 'combined/combined_methods_MS.csv'
final_df.to_csv(output_file, index=False)

print(f"Combined file saved at: {output_file}")
