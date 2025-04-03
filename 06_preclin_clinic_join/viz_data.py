import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_top_entities_side_by_side(df, id_column, condition_column, drug_column, top_n=20, color_code='#E69F00', viz_name_suffix='preclinical'):
    df = df[(df[condition_column].notna()) & (df[condition_column] != '')]
    df = df[(df[drug_column].notna()) & (df[drug_column] != '')]
    
    # Group by conditions and count unique IDs
    condition_counts = (
        df.groupby(condition_column)[id_column]
        .nunique()
        .reset_index()
        .rename(columns={id_column: 'Count'})
        .sort_values(by='Count', ascending=False)
        .head(top_n)
    )
    
    # Group by drugs and count unique IDs
    drug_counts = (
        df.groupby(drug_column)[id_column]
        .nunique()
        .reset_index()
        .rename(columns={id_column: 'Count'})
        .sort_values(by='Count', ascending=False)
        .head(top_n)
    )
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # Plot conditions
    axes[0].barh(condition_counts[condition_column], condition_counts['Count'], color=color_code, zorder=2)
    axes[0].set_title(f'Top {top_n} Conditions by {id_column} Count')
    axes[0].set_xlabel('Studies Count', fontsize=16)
    #axes[0].set_ylabel('Conditions')
    axes[0].invert_yaxis()
    for i, v in enumerate(condition_counts['Count']):
        axes[0].text(v + 0.1, i, str(v), va='center', fontsize=12)
        
    axes[0].tick_params(axis='both', labelsize=13)
    axes[0].grid(axis='x', linestyle='--', alpha=0.4, color='gray', zorder=0)

    # Plot drugs
    axes[1].barh(drug_counts[drug_column], drug_counts['Count'], color=color_code, zorder=2)
    axes[1].set_title(f'Top {top_n} Drugs by {id_column} Count')
    axes[1].set_xlabel('Studies Count', fontsize=16)
    #axes[1].set_ylabel('Drugs')
    axes[1].invert_yaxis()
    for i, v in enumerate(drug_counts['Count']):
        axes[1].text(v + 0.1, i, str(v), va='center', fontsize=12)
        
    axes[1].tick_params(axis='both', labelsize=13)
    axes[1].grid(axis='x', linestyle='--', alpha=0.4, color='gray', zorder=0)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"06_preclin_clinic_join/viz/top_{top_n}_drug_disease_distr_{viz_name_suffix}.png")
    plt.show()

def viz_joined_preclin_clinical(filtered_df, normalized_key="normalized_condition", additional_sort_by="both", translation_column=None, top_n=25, fig_name_suffix=''):
    """
    Visualizes the top N normalized conditions with preclinical and clinical counts as horizontal bars.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing the counts and normalized keys.
    - normalized_key (str): Column name for the key (e.g., condition, drug).
    - additional_sort_by (str): Sorting criterion - "both", "clinical_count", or "preclinical_count".
    - translation_column (str or None): Column name indicating whether a symbol should be added for clinical count.
    - top_n (int): Number of top records to display.

    Returns:
    - None: Displays the plot.
    """
    if additional_sort_by == "both":
        top_n_df = filtered_df.head(top_n)
        title_str = "Clinical + Preclinical Count"
    elif additional_sort_by == "clinical_count":
        top_n_df = filtered_df.sort_values(by='clinical_count', ascending=False).head(top_n)
        title_str = "Clinical Count"
    elif additional_sort_by == "tail":
        top_n_df = filtered_df.tail(top_n)
        title_str = "Clinical + Preclinical Count Tail"
    else:
        top_n_df = filtered_df.sort_values(by='preclinical_count', ascending=False).head(top_n)
        title_str = "Preclinical Count"

    # Data for plotting
    conditions = top_n_df[normalized_key]
    clinical_counts = top_n_df['clinical_count']
    preclinical_counts = top_n_df['preclinical_count']
    
    # Bar positions
    y_positions = np.arange(len(conditions))  # Position for each condition
    bar_width = 0.4  # Width of the bars
    
    # Plotting
    plt.figure(figsize=(14, 10))
    plt.barh(y_positions - bar_width / 2, preclinical_counts, height=bar_width, label='Preclinical Count', color="#56B4E9",zorder=2)
    plt.barh(y_positions + bar_width / 2, clinical_counts, height=bar_width, label='Clinical Count', color="#E69F00",zorder=2)

    # Add labels and optional symbol to each bar
    for i in range(len(conditions)):
        plt.text(preclinical_counts.iloc[i], y_positions[i] - bar_width / 2, f'{preclinical_counts.iloc[i]:.0f}', va='center',fontsize=16)
        
        clinical_text = f'{clinical_counts.iloc[i]:.0f}'
        if translation_column and top_n_df[translation_column].iloc[i]:
            clinical_text += " ♦"  # Add diamond symbol
        plt.text(clinical_counts.iloc[i], y_positions[i] + bar_width / 2, clinical_text, va='center', fontsize=16)

    # Adding labels and legend
    plt.yticks(y_positions, conditions)
    plt.xlabel('Study Count', fontsize=15)
    #plt.ylabel('Normalized Condition')
    plt.title(f'Top {top_n} Disease-Drug Pairs by {title_str}', fontsize=18)
    
    # Update legend to include diamond explanation if translation_column is provided
    if translation_column:
        plt.legend(handles=[
            plt.Line2D([0], [0], color="#56B4E9", lw=4, label='Preclinical Count'),
            plt.Line2D([0], [0], color="#E69F00", lw=4, label='Clinical Count'),
            plt.Line2D([0], [0], color="black", marker="D", linestyle='', label='At least one Phase 4 trial')
        ], loc='lower right', fontsize=16)
    else:
        plt.legend(title="Legend", loc='lower right', fontsize=16)
    plt.tick_params(axis='both', labelsize=18)
    # Adjust layout
    plt.gca().invert_yaxis()  # Reverse the order to display from largest to smallest
    plt.grid(axis='x', linestyle='--', alpha=0.4, color='gray', zorder=0)

    plt.tight_layout()
    plt.savefig(f'06_preclin_clinic_join/viz/top{top_n}_preclin_clin{fig_name_suffix}.pdf')
    plt.show()
    
output_path = f"06_preclin_clinic_join/data/manual_data_checks/condition_clinical_and_preclinical_13607.csv"
filtered_df = pd.read_csv(output_path)

# ------------------------- #
#         VISUALIZE         #
# ------------------------- #

viz_joined_preclin_clinical(
    filtered_df,
    "normalized_key",
    translation_column='at_least_one_phase4',
    top_n=15,
    fig_name_suffix='_disease_drug'
)