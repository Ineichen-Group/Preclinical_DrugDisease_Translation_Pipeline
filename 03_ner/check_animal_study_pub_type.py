import os
import pandas as pd

def filter_and_count_by_keywords(
    df,
    column,
    keywords,
    output_dir=None,
    output_prefix='',
    save=True
):
    """
    Filters a DataFrame by checking if given keywords are contained in a specified column.
    Saves filtered subsets and a count summary if requested.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column to search in.
    - keywords (list of str): List of case-insensitive strings to match.
    - output_dir (str): Directory to save outputs (optional).
    - output_prefix (str): Prefix for output filenames (optional).
    - save (bool): Whether to save filtered CSVs and count summary.

    Returns:
    - counts_df (pd.DataFrame): Summary of counts by keyword.
    - filtered_dfs (dict): Dictionary of filtered DataFrames by keyword.
    """

    filtered_dfs = {}
    counts = []

    for keyword in keywords:
        filtered_df = df[df[column].str.contains(keyword, case=False, na=False)]
        filtered_dfs[keyword] = filtered_df
        counts.append({'type': keyword, 'count': filtered_df.shape[0]})

        if save and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_prefix}{keyword.replace(' ', '_').lower()}_publications.csv"
            filtered_df.to_csv(os.path.join(output_dir, filename), index=False)

    counts_df = pd.DataFrame(counts)

    if save and output_dir:
        summary_path = os.path.join(output_dir, f"{output_prefix}publication_type_counts.csv")
        counts_df.to_csv(summary_path, index=False)

        print("Saved:")
        for keyword in keywords:
            print(f"- {filtered_dfs[keyword].shape[0]} rows for '{keyword}'")
        print(f"- Summary to '{summary_path}'")

    return counts_df, filtered_dfs

df = pd.read_csv("03_ner/data/animal_studies_with_drug_disease/animal_studies_metadata_595768.csv")

counts_df, filtered_dfs = filter_and_count_by_keywords(
    df,
    column='publication_type',
    keywords=['review', 'clinical trial', 'case report'],
    output_dir='03_ner/check_study_type',
    output_prefix='animal_studies_'
)
