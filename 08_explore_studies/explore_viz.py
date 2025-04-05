import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
import numpy as np

path_linked = "06_preclin_clinic_join/data/joined_data/condition_clinical_and_preclinical_13023.csv"
path_clin_metadata = "06_preclin_clinic_join/data/joined_data/clinical_metadata_mapped.csv"
path_preclin_metadata = "06_preclin_clinic_join/data/joined_data/preclinical_metadata_mapped.csv"

mapped_studies = pd.read_csv(path_linked)
df_clinical_metadata = pd.read_csv(path_clin_metadata)
df_preclinical_metadata = pd.read_csv(path_preclin_metadata)

disease = "multiple sclerosis"
drug = "show all"

# APPLY FILTERS 
if disease != "show all" and drug != "show all":
    drug_pattern = r"\b({})\b".format(drug)
    disease_pattern = r"\b({})\b".format(disease)
    mapped_studies = mapped_studies[
        mapped_studies["disease"]
        .str.contains(disease_pattern, case=False, na=False, flags=re.IGNORECASE)
        & mapped_studies["drug"]
        .str.contains(drug_pattern, case=False, na=False, flags=re.IGNORECASE)
    ]
elif disease != "show all":
    disease_pattern = r"\b({})\b".format(disease)
    mapped_studies = mapped_studies[
        mapped_studies["disease"]
        .str.contains(disease_pattern, case=False, na=False, flags=re.IGNORECASE)
    ]
elif drug != "show all":
    drug_pattern = r"\b({})\b".format(drug)
    mapped_studies = mapped_studies[
        mapped_studies["drug"]
        .str.contains(drug_pattern, case=False, na=False, flags=re.IGNORECASE)
    ]
    
# GET META VIEW
mapped_studies['clinical_doc_ids'] = mapped_studies['clinical_doc_ids'].apply(ast.literal_eval)
mapped_studies['preclinical_doc_ids'] = mapped_studies['preclinical_doc_ids'].apply(ast.literal_eval)

# Flatten all clinical_doc_ids (NCTIDs) into a set
all_nctids = set(nctid for sublist in mapped_studies['clinical_doc_ids'] for nctid in sublist)
# Flatten all preclinical_doc_ids (PMIDs) into a set
all_pmids = set(pmid for sublist in mapped_studies['preclinical_doc_ids'] for pmid in sublist)

all_diseases = set(mapped_studies['disease'])
all_drugs = set(mapped_studies['drug'])

print(f"Unique clinical with mapping: {len(all_nctids)}, preclinical: {len(all_pmids)}")
print(f"Unique diseases: {len(all_diseases)}, drugs: {len(all_drugs)}")

# Group by drug and sum up the counts
drug_counts = mapped_studies.groupby("drug")[["preclinical_count", "clinical_count"]].sum()

# Sort by total count (preclinical + clinical), descending
drug_counts["total_count"] = drug_counts["preclinical_count"] + drug_counts["clinical_count"]
drug_counts = drug_counts.sort_values("total_count", ascending=False)

# Take top 50
drug_counts = drug_counts.head(50)

# Create the bar chart
plt.figure(figsize=(14, 10))

# Data for plotting
bar_width = 0.4
y_positions = np.arange(len(drug_counts))
preclinical_counts = drug_counts["preclinical_count"]
clinical_counts = drug_counts["clinical_count"]

# Plotting bars
plt.barh(y_positions - bar_width / 2, preclinical_counts, height=bar_width, label="Preclinical Count", zorder=2)
plt.barh(y_positions + bar_width / 2, clinical_counts, height=bar_width, label="Clinical Count", zorder=2)

# Add labels to bars
for i in range(len(drug_counts)):
    plt.text(preclinical_counts.iloc[i], y_positions[i] - bar_width / 2, f'{preclinical_counts.iloc[i]:.0f}', va='center', fontsize=12)
    plt.text(clinical_counts.iloc[i], y_positions[i] + bar_width / 2, f'{clinical_counts.iloc[i]:.0f}', va='center', fontsize=12)

# Final plot formatting
plt.yticks(y_positions, drug_counts.index)
plt.xlabel("Study Count", fontsize=15)
plt.title("Preclinical vs Clinical Trial Counts per Drug", fontsize=16)
plt.gca().invert_yaxis()
plt.legend(fontsize=14)
plt.grid(axis='x', linestyle='--', zorder=1)
plt.tight_layout()
plt.show()

df_to_show = mapped_studies[['disease', 'drug', 'clinical_count','preclinical_count','max_phase']]


# GET CLINICAL VIEW
df_clinical_metadata = df_clinical_metadata[df_clinical_metadata['nct_id'].isin(all_nctids)]

# GET PRECLINICAL VIEW
df_preclinical_metadata = df_preclinical_metadata[df_preclinical_metadata['PMID'].isin(all_pmids)]

# STUDIES OVER TIME
preclinical_by_year = df_preclinical_metadata["year"].value_counts().sort_index().astype(int)
clinical_by_year = df_clinical_metadata["start_year"].value_counts().sort_index().astype(int)

# Combine into one DataFrame and align years
study_counts = (
    pd.DataFrame({
        "preclinical_count": preclinical_by_year,
        "clinical_count": clinical_by_year
    })
    .fillna(0)
    .astype(int)
)

# Sort by year
study_counts = study_counts.sort_index()
study_counts.index = study_counts.index.astype(int)

# Plotting
plt.figure(figsize=(14, 8))
bar_width = 0.4
x = np.arange(len(study_counts))

# Bar plots
plt.bar(x - bar_width / 2, study_counts["preclinical_count"], width=bar_width, label="Preclinical Count", zorder=2)
plt.bar(x + bar_width / 2, study_counts["clinical_count"], width=bar_width, label="Clinical Count", zorder=2)

# Annotate bar values
for i in range(len(study_counts)):
    plt.text(x[i] - bar_width / 2, study_counts["preclinical_count"].iloc[i] + 0.5,
             f'{study_counts["preclinical_count"].iloc[i]}', ha='center', fontsize=9)
    plt.text(x[i] + bar_width / 2, study_counts["clinical_count"].iloc[i] + 0.5,
             f'{study_counts["clinical_count"].iloc[i]}', ha='center', fontsize=9)

# Formatting
plt.xticks(x, study_counts.index, rotation=45)
plt.ylabel("Number of Studies", fontsize=14)
plt.xlabel("Year", fontsize=14)
plt.title("Number of Preclinical vs Clinical Studies Over Time", fontsize=16)
plt.legend()
plt.grid(axis='y', linestyle='--', zorder=1)
plt.tight_layout()
plt.show()