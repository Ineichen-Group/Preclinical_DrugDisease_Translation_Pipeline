import pandas as pd
import random

# Read CSV
df = pd.read_csv(
    "04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_595768.csv"
)

random.seed(42)

def explode_column(df, colname):
    rows = []
    for _, row in df.iterrows():
        pmid = row["PMID"]
        if pd.notna(row[colname]):
            for item in row[colname].split("|"):
                rows.append({"PMID": pmid, colname: item.strip()})
    return pd.DataFrame(rows)

col_name_drugs = "linkbert_mapped_drugs"
col_name_disease = "linkbert_mapped_conditions"

exploded_conditions = explode_column(df, col_name_disease)
exploded_drugs = explode_column(df, col_name_drugs)

# === 1) Sample 100 diseases (unchanged) ===
sampled_conditions = (
    exploded_conditions
    .drop_duplicates(subset=col_name_disease)
    .sample(
        n=min(100, exploded_conditions[col_name_disease].nunique()),
        random_state=42,
    )
)

# === 2) Sample base 100 drugs (no special filtering) ===
sampled_drugs_100 = (
    exploded_drugs
    .drop_duplicates(subset=col_name_drugs)
    .sample(
        n=min(100, exploded_drugs[col_name_drugs].nunique()),
        random_state=42,
    )
)

# === 3) Build filtered pool for extra 50 ===
ban_words = ["agent", "compounds", "hormone", "factor", "activator", "molecule", "inhibition", "conjugate", "antagonism", "channel", "antagonist", "inhibitor", "receptor", "antibody", "peptide", "blocker", "agonist", "analogue", "modulator", "derivative", "ligand"]

def contains_banned(text):
    text_l = str(text).lower()
    return any(b in text_l for b in ban_words)

filtered_drugs = exploded_drugs[
    ~exploded_drugs[col_name_drugs].apply(contains_banned)
]

# Remove drugs already used in the first 100 (by name)
filtered_drugs_extra_pool = filtered_drugs[
    ~filtered_drugs[col_name_drugs].isin(sampled_drugs_100[col_name_drugs])
]

# Drop duplicates on drug name so each appears once
filtered_drugs_extra_pool = filtered_drugs_extra_pool.drop_duplicates(subset=col_name_drugs)

# === 4) Sample extra 50 from remaining pool ===
extra_50_drugs = filtered_drugs_extra_pool.sample(
    n=min(50, filtered_drugs_extra_pool[col_name_drugs].nunique()),
    random_state=42,
)

# === 5) Combine to 150 drugs, ensure no overlap ===
combined_150_drugs = pd.concat(
    [sampled_drugs_100, extra_50_drugs],
    ignore_index=True
)

# Sanity check: ensure no duplicate drug names
combined_150_drugs = combined_150_drugs.drop_duplicates(subset=col_name_drugs)
print("Unique drugs in combined set:", combined_150_drugs[col_name_drugs].nunique())

# === 6) Save outputs ===
sampled_conditions.to_csv(
    "04_normalization/data/ner_samples/sampled_conditions.csv",
    index=False,
)
sampled_drugs_100.to_csv(
    "04_normalization/data/ner_samples/sampled_drugs_100.csv",
    index=False,
)
extra_50_drugs.to_csv(
    "04_normalization/data/ner_samples/extra_50_drugs_filtered.csv",
    index=False,
)
combined_150_drugs.to_csv(
    "04_normalization/data/ner_samples/combined_150_drugs.csv",
    index=False,
)
