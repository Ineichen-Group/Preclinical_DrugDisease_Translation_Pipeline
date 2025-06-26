from core.term_dict_mapper import generate_conditions_lookup_dictionary
from core.term_dict_mapper import lookup_canonical, process_dataframe
from core.term_dict_mapper import generate_drugs_lookup_dictionary
import pandas as pd

def process_conditions(df_pred_unique, term_file_path="04_normalization/data/term_dictionaries/diseases_dictionary_mesh_icd_2024.csv", condition_column="unique_conditions_linkbert_predictions"):
    """
    Maps medical conditions to standardized terms using MeSH/ICD dictionaries.
    
    Takes a DataFrame with predicted conditions and normalizes them against a 
    medical terminology dictionary, then generates mapping statistics.
    
    Args:
        df_pred_unique (pd.DataFrame): DataFrame containing unique condition predictions
        term_file_path (str): Path to the medical terminology dictionary CSV file
        condition_column (str): Name of the column containing condition predictions
        output_stats_path (str): Path where mapping statistics CSV will be saved
    
    Returns:
        pd.DataFrame: DataFrame with conditions mapped to canonical terms
        
    Side Effects:
        - Saves mapping statistics to the specified output path
    """
    synonyms_dict = generate_conditions_lookup_dictionary(term_file_path)

    df_conditions_mapped, match_counts, processed_counts, same_condition_counts = process_dataframe(
        df_pred_unique, synonyms_dict, condition_column
    )

    # Generate mapping statistics
    counts_df_condition = pd.DataFrame({
        'Processed Condition': processed_counts,
        'Matched Condition': match_counts,
        #'Same Condition Counts': same_condition_counts
    })

    counts_df_condition['% Mapped Condition'] = round(
        (counts_df_condition['Matched Condition'] / counts_df_condition['Processed Condition']) * 100, 2
    )
    counts_df_condition.reset_index(inplace=True)
    counts_df_condition.rename(columns={'index': 'Annotations Source'}, inplace=True)
    
    output_stats_path = f"04_normalization/nen_stats/condition_dict_mapped_{len(df_pred_unique)}.csv"
    counts_df_condition.to_csv(output_stats_path, index=False)

    return df_conditions_mapped

def process_interventions(df_conditions_mapped, drug_column="unique_interventions_linkbert_predictions"):
    path_prefix = "04_normalization/data/term_dictionaries"
    file_configs = [
        {
            'path': path_prefix + "/drug_names_terminology/drugs_dictionary_medlineplus.csv",
            'id_col': 0,
            'name_col': 1,
            'synonym_col': 2,
            'delimiter': ','
        },
        {
            'path': path_prefix + "/drug_names_terminology/drugs_dictionary_nhs.csv",
            'id_col': 0,
            'name_col': 1,
            'synonym_col': 2,
            'delimiter': ','
        },
        {
            'path': path_prefix + "/drug_names_terminology/drugbank_vocabulary.csv",
            'id_col': 0,
            'name_col': 2,
            'synonym_col': 5,
            'delimiter': ','
        },
        {
            'path': path_prefix + "/drug_names_terminology/drugs_dictionary_mesh.csv",
            'id_col': 0,
            'name_col': 1,
            'synonym_col': 2,
            'delimiter': ','
        },
        {
            'path': path_prefix + "/drug_names_terminology/drugs_dictionary_wikipedia.csv",
            'id_col': 0,
            'name_col': 1,
            'synonym_col': 2,
            'delimiter': ','
        }
    ]

    drug_variant_to_canonical, drug_canonical_to_data = generate_drugs_lookup_dictionary(file_configs)

    df_drugs_mapped, match_counts, processed_counts, same_condition_counts = process_dataframe(df_conditions_mapped, drug_variant_to_canonical, drug_column, "drugs")

    counts_df_interventions = pd.DataFrame({
        'Processed Intervention': processed_counts,
        'Matched Intervention': match_counts,
        #'Same Condition Counts': same_condition_counts
    })

    counts_df_interventions['% Mapped Intervention'] = round((counts_df_interventions['Matched Intervention'] / counts_df_interventions['Processed Intervention']) * 100, 2)
    counts_df_interventions.reset_index(inplace=True)
    counts_df_interventions.rename(columns={'index': 'Annotations Source'}, inplace=True)
    output_stats_path = f"04_normalization/nen_stats/intervention_dict_mapped_{len(df_conditions_mapped)}.csv"
    counts_df_interventions.to_csv(output_stats_path, index=False)

    return df_drugs_mapped


def main():
    ner_outputs_path = "03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_4489.csv" #"03_IE_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_595768.csv"
    df_pred_unique = pd.read_csv(ner_outputs_path)
    
    df_conditions_mapped = process_conditions(df_pred_unique)
    print(f"Shape after disease mapping: {df_conditions_mapped.shape}")
    
    df_condition_interventions_mapped = process_interventions(df_conditions_mapped)
    print(f"Shape after drug mapping: {df_condition_interventions_mapped.shape}")

    df_condition_interventions_mapped.to_csv(f"04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_{len(df_condition_interventions_mapped)}.csv")

 
if __name__ == "__main__":
    main()