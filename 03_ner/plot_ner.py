import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def plot_entity_distribution_by_type(
    df_pred,
    entity_type,
    pred_col_name='ner_prediction_BioLinkBERT-base_normalized',
    doc_id_col='PMID',
    top_n=20,
    save_entity_counts_to=None,
    save_plot_to=None
):
    """
    Extracts statistics and plots the distribution of the most frequent entities
    of a given type (e.g., DRUG, DISEASE) from a specified NER prediction column.

    Args:
    - df_pred (DataFrame): DataFrame containing a column with NER predictions.
    - entity_type (str): Entity type to filter by (e.g., 'DRUG', 'DISEASE').
    - pred_col_name (str): Column name containing the NER prediction data.
    - doc_id_col (str): Column name for document IDs (e.g., 'PMID', 'doc_id').
    - top_n (int): Number of most frequent entities to display.
    - save_entity_counts_to (str): Folder path to save CSV of entity counts (optional).
    - save_plot_to (str): Full file path (with .png) to save the plot (optional).

    Returns:
    - None: Displays a bar plot and optionally saves a CSV of entity counts and a PNG of the plot.
    """

    entity_list = []
    entity_to_docs = defaultdict(set)
    doc_id_set = set()

    for _, row in df_pred.iterrows():
        doc_id = row[doc_id_col]
        doc_id_set.add(doc_id)

        try:
            ner_predictions = eval(row[pred_col_name])
        except Exception as e:
            print(f"Error parsing predictions for doc {doc_id}: {e}")
            continue

        for entity in ner_predictions:
            if len(entity) < 4:
                continue
            ent_type = entity[2]
            ent_text = entity[3]
            if ent_type == entity_type:
                entity_list.append(ent_text)
                entity_to_docs[ent_text].add(doc_id)

    # Count entity frequencies
    entity_counter = Counter(entity_list)
    entity_df = pd.DataFrame(entity_counter.items(), columns=[entity_type.title(), 'Count']) \
                   .sort_values(by='Count', ascending=False)

    # Save CSV if requested
    if save_entity_counts_to:
        os.makedirs(save_entity_counts_to, exist_ok=True)
        detailed_df = pd.DataFrame([
            {
                entity_type.title(): ent,
                'Count': len(docs),
                'Documents': ", ".join(map(str, sorted(docs)))
            }
            for ent, docs in entity_to_docs.items()
        ]).sort_values(by='Count', ascending=False)
        csv_path = os.path.join(save_entity_counts_to, f"{entity_type.lower()}_names_counts_{len(doc_id_set)}.csv")
        detailed_df.to_csv(csv_path, index=False)
        print(f"Saved entity counts to {csv_path}")

    # Plot top N
    entity_df = entity_df.head(top_n)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(entity_df[entity_type.title()], entity_df['Count'], color='#56B4E9')
    plt.title(f'Most Frequent {entity_type.title()} Entities', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(f'{entity_type.title()} Entities')
    plt.gca().invert_yaxis()

    # Add count labels
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{int(bar.get_width())}', va='center')

    plt.tight_layout()

    # Save plot if requested
    if save_plot_to:
        os.makedirs(os.path.dirname(save_plot_to), exist_ok=True)
        plt.savefig(save_plot_to)
        print(f"Plot saved to {save_plot_to}")

    plt.show()


def plot_drug_disease_distribution(df_pred, pred_col_name='ner_prediction_BioLinkBERT-base_normalized', top_n=20, save_drug_disease_counts_to=None):
    """
    Extracts statistics and plots the distribution of most frequent DRUG and DISEASE entities
    from the ner_prediction_BioLinkBERT-base_normalized column.
    
    Args:
    - df_pred (DataFrame): A DataFrame containing a 'ner_prediction_BioLinkBERT-base_normalized' column
                           with entity tuples.
    - top_n (int): The number of most frequent entities to plot (default: 20).
    
    Returns:
    - None: Plots the distributions of the most frequent DRUG and DISEASE entities.
    """

    # Initialize lists for drug and disease entities
    drug_entities = []
    disease_entities = []

    drug_dict = defaultdict(set)  # Store PMIDs for each drug
    disease_dict = defaultdict(set)  # Store PMIDs for each disease
    pmids_set = set()

    # Extract entities from the dataframe
    for index, row in df_pred.iterrows():
        pmid = row['PMID']  # Extract PMID
        pmids_set.add(pmid)
        # Evaluate the string representation to convert into list
        ner_predictions = eval(row[pred_col_name])

        # Loop through the entities and classify as DRUG or DISEASE
        for entity in ner_predictions:
            entity_type = entity[2]
            entity_text = entity[3]
            if entity_type == 'DRUG':
                drug_entities.append(entity_text)
                drug_dict[entity_text].add(pmid)
            elif entity_type == 'DISEASE':
                disease_entities.append(entity_text)
                disease_dict[entity_text].add(pmid)  # Add PMID to disease entry

    # Count the occurrences of each entity
    drug_counter = Counter(drug_entities)
    disease_counter = Counter(disease_entities)

    # Okabe-Ito color palette
    okabe_ito_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']

    # Convert counters to dataframes for easy manipulation and select top N entities
    drug_df = pd.DataFrame(drug_counter.items(), columns=['Drug', 'Count']).sort_values(by='Count', ascending=False)
    disease_df = pd.DataFrame(disease_counter.items(), columns=['Disease', 'Count']).sort_values(by='Count', ascending=False)

    if save_drug_disease_counts_to:
        drug_df_save = pd.DataFrame([
            {'Drug': drug, 'Count': len(pmids), 'PMIDs': ", ".join(map(str, pmids))}
            for drug, pmids in drug_dict.items()
        ]).sort_values(by='Count', ascending=False)
        print(f'Unique drugs count {len(drug_dict)}')
        disease_df_save = pd.DataFrame([
            {'Disease': disease, 'Count': len(pmids), 'PMIDs': ", ".join(map(str, pmids))}
            for disease, pmids in disease_dict.items()
        ]).sort_values(by='Count', ascending=False)
        print(f'Unique disease count {len(disease_dict)}')
        drug_df_save.to_csv(f"{save_drug_disease_counts_to}/drug_names_counts_{len(pmids_set)}.csv", index=False)
        disease_df_save.to_csv(f"{save_drug_disease_counts_to}/disease_names_counts_{len(pmids_set)}.csv", index=False)

    drug_df = drug_df.head(top_n)
    disease_df = disease_df.head(top_n)

    # Plotting distributions
    plt.figure(figsize=(12, 6))

    # Drug entity distribution
    plt.subplot(1, 2, 1)
    bars = plt.barh(drug_df['Drug'], drug_df['Count'], color=okabe_ito_colors[0])
    plt.title('Most Frequent Drug Entities')
    plt.xlabel('Count')
    plt.ylabel('Drug Entities')
    plt.gca().invert_yaxis()  # Invert y-axis to show most frequent on top

    # Add counts at the end of the bars for drugs
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width()}', va='center')

    # Disease entity distribution
    plt.subplot(1, 2, 2)
    bars = plt.barh(disease_df['Disease'], disease_df['Count'], color=okabe_ito_colors[1])
    plt.title('Most Frequent Disease Entities')
    plt.xlabel('Count')
    plt.ylabel('Disease Entities')
    plt.gca().invert_yaxis()  # Invert y-axis to show most frequent on top

    # Add counts at the end of the bars for diseases
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width()}', va='center')

    plt.tight_layout()
    plt.show()
    

# ------------------------- #
#         VISUALIZE         #
# ------------------------- #
entity_type = 'STRAIN'
file_path = f"./03_ner/model_predictions/strain/test_annotated_BioLinkBERT-base_tuples_20250416_part_1.csv"
filtered_df = pd.read_csv(file_path)

plot_entity_distribution_by_type(
    filtered_df,
    entity_type=entity_type,
    doc_id_col='doc_id',
    top_n=25,
    save_entity_counts_to='ner_stats/entity_counts',
    save_plot_to=f'./03_ner/viz/{entity_type.lower()}_distribution.png'
)
