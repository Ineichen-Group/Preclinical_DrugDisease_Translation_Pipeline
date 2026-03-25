import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import re
import csv
import os
import glob
from plot_ner import plot_drug_disease_distribution
from abbreviations import schwartz_hearst

# Function to remove spaces around ' and -
def remove_spaces_around_apostrophe_and_dash(text):
    text = text.replace(" ' ", "'")  # Remove spaces around '
    text = text.replace("' s", "'s")  # Remove spaces around '
    text = text.replace(" - ", "-")  # Remove spaces around -
    text = text.replace("- ", "-")  # Remove spaces around -
    text = text.replace(" / ", "/")  # Remove spaces around /
    text = text.replace("( ", "(")  # Remove spaces around (
    text = text.replace(" )", ")")  # Remove spaces around -
    text = text.replace("[ ", "[")  # Remove spaces around [
    text = text.replace(" ]", "]")  # Remove spaces around ]
    text = re.sub(r'\s+', ' ', text)  # normalize spaces
    return text

def process_ner_predictions(directory_path):
    # Initialize an empty list to store DataFrames
    dfs = []

    count_files = 0
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            # Load the CSV into a DataFrame and select specific columns
            df = pd.read_csv(file_path)[['PMID', 'ner_prediction_BioLinkBERT-base_normalized']]
            # Append the DataFrame to the list
            dfs.append(df)
            count_files += 1

    # Concatenate all DataFrames in the list
    df_pred_full = pd.concat(dfs, ignore_index=True).drop_duplicates()

    # Drop documents with no extracted entities
    df_empty = df_pred_full[df_pred_full["ner_prediction_BioLinkBERT-base_normalized"].apply(lambda x: len(x) <= 2)]
    df_pred = df_pred_full[df_pred_full["ner_prediction_BioLinkBERT-base_normalized"].apply(lambda x: len(x) > 2)]
    df_pred['ner_prediction_BioLinkBERT-base_normalized'] = df_pred['ner_prediction_BioLinkBERT-base_normalized'].apply(remove_spaces_around_apostrophe_and_dash)
    print(f"Read {count_files} number of files, full df shape {df_pred_full.shape}, df shape without empty NER {df_pred.shape}")
    return df_pred, df_empty

def extract_abbreviation_from_full_text(pmid_set, folder_path = "data/animal_studies_for_ner_inference", save_to_path="03_IE_ner/data/abbreviations_expansion/pmid_abbreviations.csv"):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    count_files = 0
    # Loop through CSV files and filter data
    for file in csv_files:
        if count_files in [100, 200, 300, 400, 500, 600]:
            print(f'processing reached {count_files} with {file}')
        count_files += 1
        df = pd.read_csv(file)  # Read the CSV file
        if "PMID" in df.columns:  # Ensure the column exists
            filtered_df = df[df["PMID"].isin(pmid_set)]  # Faster lookup using a set
            filtered_df = filtered_df.copy()
            filtered_df['abbreviation_definition_pairs'] = filtered_df['Text'].apply(extract_abbreviation_definition_pairs)
            if not filtered_df.empty:
                if count_files == 0:
                    filtered_df[['PMID', 'abbreviation_definition_pairs']].to_csv(save_to_path, index=False)
                else:
                    filtered_df[['PMID', 'abbreviation_definition_pairs']].to_csv(save_to_path, mode='a', header=False, index=False)
    print("Completed reading full text docs.")
    return f'{save_to_path}/pmid_abbreviations.csv'

def extract_abbreviation_definition_pairs(doc_text):
    pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=doc_text)
    return pairs

def extract_unique_entities(nct_id, annotation_list, abbreviation_definition_pairs, model="linkbert"):
    unique_conditions = set()
    unique_interventions = set()
    interventions_type = set()
   
    try:
        annotation_list = eval(annotation_list)
    except SyntaxError as e:
        print(nct_id)
        print(annotation_list)
        print("Syntax error in eval:", e)
        return "issues processing line"
    
    for annotation in annotation_list:
        _, _, entity_type, entity_name = annotation
        if entity_name.startswith("##"):
            continue ## NEED TO INVESTIGATE
        if not entity_name or len(entity_name) == 1:
            continue ## ASSUME TOKENIZER ERROR 
            
        # REPLACE ABBREVIATIONS WITH FULL FORM
        if entity_name in abbreviation_definition_pairs:
            #print("Skipping entity {} as it is an ABBR".format(entity_name))
            entity_name = abbreviation_definition_pairs[entity_name] 
            #continue
        if entity_name.upper() in abbreviation_definition_pairs:
            #print("Skipping entity {} as it is an ABBR".format(entity_name))
            entity_name = abbreviation_definition_pairs[entity_name.upper()] 
        entity_name = entity_name.lower()
        if entity_type == 'DISEASE':
            unique_conditions.add(entity_name)
        elif entity_type == 'DRUG':
            unique_interventions.add(entity_name)
            interventions_type.add(entity_type)
        
    return "|".join(list(unique_conditions)), "|".join(list(unique_interventions)), "|".join(list(interventions_type))

def get_emtpy_ner_stats(df_pred):
    # Count rows where unique_interventions_linkbert_predictions is empty
    empty_interventions = df_pred["unique_interventions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()

    # Count rows where unique_conditions_linkbert_predictions is empty
    empty_conditions = df_pred["unique_conditions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and not x).sum()

    # Count rows where both are empty
    both_empty = df_pred[(df_pred["unique_conditions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and not x)) &
                    (df_pred["unique_interventions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and x.strip() == ""))].shape[0]
    results_empty_entities = pd.DataFrame({
        "Empty unique_conditions_linkbert_predictions": [empty_conditions],
        "Empty unique_interventions_linkbert_predictions": [empty_interventions],
        "Both Empty": [both_empty]
    })
    print(results_empty_entities)
    results_empty_entities.to_csv(f"03_IE_ner/ner_stats/empty_ner_predictions_count_{len(results_empty_entities)}.csv")
    
def load_abbreviations_from_csv(save_abbrev_to_path, pmid_set, full_text_dir="02_animal_study_classification/data/animal_studies_for_ner"):
    if not os.path.isfile(save_abbrev_to_path):
        print(f"Abbreviations file not found at {save_abbrev_to_path}")
        print(f"Extracting abbreviations for {len(pmid_set)} PMIDs...")
        extract_abbreviation_from_full_text(pmid_set, full_text_dir, save_abbrev_to_path)
        print(f"Abbreviations saved to {save_abbrev_to_path}")
    else:
        print(f"Loading existing abbreviations from {save_abbrev_to_path}")
    
    abbrev_df = pd.read_csv(save_abbrev_to_path, names=["PMID", "abbreviation_definition_pairs"])
    abbrev_df["abbreviation_definition_pairs"] = abbrev_df["abbreviation_definition_pairs"].apply(eval)
    
    print(f"Loaded abbreviations for {len(abbrev_df)} PMIDs")
    return abbrev_df
    

def main():
    df_pred, df_empty = process_ner_predictions("03_IE_ner/model_predictions/disease_from_model_regex")
    #plot_drug_disease_distribution(df_pred, save_drug_disease_counts_to="03_ner/ner_stats")
    pmid_set = set(df_pred["PMID"])
    
    save_abbrev_to_path = f"03_IE_ner/data/abbreviations_expansion/pmid_abbreviations_{len(pmid_set)}.csv"
    abbrev_df = load_abbreviations_from_csv(save_abbrev_to_path, pmid_set)
    df_pred_with_abbrev = df_pred.merge(abbrev_df, on="PMID", how="left")  # Left join to keep all df_main rows
    print(f"abbrev {abbrev_df.shape}, joined {df_pred_with_abbrev.shape}")
    
    # Extract unique condition and intervention predictions per article
    print("Extracting unique entities")
    model_name_str_biolink = "linkbert"
    biolinkbert_col = "ner_prediction_BioLinkBERT-base_normalized"
    df_pred_with_abbrev[f'unique_conditions_{model_name_str_biolink}_predictions'], df_pred_with_abbrev[f'unique_interventions_{model_name_str_biolink}_predictions'], _ = zip(*df_pred_with_abbrev.apply(lambda row: extract_unique_entities(row['PMID'], row[biolinkbert_col], row['abbreviation_definition_pairs']), axis=1))

    get_emtpy_ner_stats(df_pred_with_abbrev)
    
    # Keep articles with both conditions and interventions
    filtered_df_non_empty = df_pred_with_abbrev[
    df_pred_with_abbrev["unique_conditions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and bool(x)) & 
    df_pred_with_abbrev["unique_interventions_linkbert_predictions"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    ]
    print(f"articles with both conditions and interventions:{filtered_df_non_empty.shape}")

    
    # save all articles
    save_dir = "03_IE_ner/data"
    df_to_save = filtered_df_non_empty[["PMID", "unique_conditions_linkbert_predictions", "unique_interventions_linkbert_predictions"]]
    df_to_save = df_to_save.drop_duplicates()
    print(f"df_to_save shape after dropping duplicates: {df_to_save.shape}")
    save_file_name = f"filtered_df_non_empty_{len(df_to_save)}"

    
    df_to_save.to_csv(f'{save_dir}/animal_studies_with_drug_disease/{save_file_name}.csv', index=False)
    df_to_save[["PMID"]].to_csv(
    f"{save_dir}/animal_studies_with_drug_disease/{save_file_name}_PMIDs.csv",
    index=False
    )
 
if __name__ == "__main__":
    main()