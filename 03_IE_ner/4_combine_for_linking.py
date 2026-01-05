import pandas as pd

file_1 = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_2879.csv"
file_2 = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_595768.csv"
file_3 = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_4489.csv"
file_4 = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_MS_5380.csv"
file_5 = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_sclerosis_10043.csv"

# Load the CSV files
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)       
df4 = pd.read_csv(file_4)
df5 = pd.read_csv(file_5)

# Combine the DataFrames
combined_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
print(f"Combined DataFrame shape: {combined_df.shape}") 

# remove duplicate rows if any based on PMID
combined_df = combined_df.drop_duplicates(subset=['PMID'])
print(f"Combined DataFrame shape after removing duplicates: {combined_df.shape}") 

#
pmids_to_keep_file = pd.read_csv("03_IE_ner/data/animal_studies_with_drug_disease/animal_studies_metadata_after_stype_filter_554307_PMIDs.csv")
pmids_to_keep = set(pmids_to_keep_file['PMID'].astype(str).tolist())    
print(f"Number of PMIDs to keep: {len(pmids_to_keep)}")
# Filter the combined DataFrame to keep only rows with PMIDs in pmids_to_keep
combined_df = combined_df[combined_df['PMID'].astype(str).isin(pmids_to_keep)]
print(f"Combined DataFrame shape after filtering PMIDs: {combined_df.shape}")
  
# Save the combined DataFrame to a new CSV file
output_file = "03_IE_ner/data/animal_studies_with_drug_disease/animal_combined_filtered_df_non_empty_ner.csv"
combined_df.to_csv(output_file, index=False)
print(f"Combined DataFrame saved to {output_file}")