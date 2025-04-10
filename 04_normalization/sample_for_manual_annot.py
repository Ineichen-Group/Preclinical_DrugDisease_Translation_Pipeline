import pandas as pd
import random

# Read CSV into pandas DataFrame
df = pd.read_csv("04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_595768.csv")

# Make sampling deterministic
random.seed(42)

# Function to explode the '|' entries while keeping PMID and column name
def explode_column(df, colname):
    rows = []
    for _, row in df.iterrows():
        pmid = row['PMID']
        if pd.notna(row[colname]):
            for item in row[colname].split('|'):
                rows.append({
                    'PMID': pmid,
                    f'{colname}': item.strip()
                })
    return pd.DataFrame(rows)

# Explode both conditions and drugs
col_name_drugs = 'linkbert_mapped_drugs'
col_name_disease= 'linkbert_mapped_conditions'

exploded_conditions = explode_column(df, col_name_disease)
exploded_drugs = explode_column(df, col_name_drugs )

# Drop duplicate entities and sample 100 from each
sampled_conditions = exploded_conditions.drop_duplicates(subset=col_name_disease).sample(n=min(100, len(exploded_conditions[col_name_disease].unique())), random_state=42)
sampled_drugs = exploded_drugs.drop_duplicates(subset=col_name_drugs).sample(n=min(100, len(exploded_drugs[col_name_drugs].unique())), random_state=42)

# Save to CSV
sampled_conditions.to_csv('04_normalization/data/ner_samples/sampled_conditions.csv', index=False)
sampled_drugs.to_csv('04_normalization/data/ner_samples/sampled_drugs.csv', index=False)