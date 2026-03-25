import os
import pandas as pd
import re

def filter_and_count_by_keywords(
    df,
    column,
    keywords,
    output_dir=None,
    output_prefix='',
    save=True,
    save_after_filter=False,
    filtered_path='./'
):
    """
    Filters a DataFrame by checking if given keywords are contained in a specified column.
    Ensures that each row is only matched by one keyword, in the order they are given.
    Optionally saves the remaining rows (not matched by any keyword) to a separate file.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Column to search in.
    - keywords (list of str): List of case-insensitive strings to match.
    - output_dir (str): Directory to save outputs (optional).
    - output_prefix (str): Prefix for output filenames (optional).
    - save (bool): Whether to save filtered CSVs and count summary.
    - save_after_filter (bool): Whether to save the remaining (non-matching) rows.
    - after_filter_filename (str): Output filename for the remaining rows.

    Returns:
    - counts_df (pd.DataFrame): Summary of counts by keyword.
    - filtered_dfs (dict): Dictionary of filtered DataFrames by keyword.
    - remaining_df (pd.DataFrame): DataFrame of unmatched rows.
    """

    filtered_dfs = {}
    counts = []

    remaining_df = df.copy()
    # drop duplicates based on PMID to avoid counting the same study multiple times
    if "PMID" in remaining_df.columns:
        remaining_df = remaining_df.drop_duplicates(subset=['PMID'])
    print(f"Starting with {remaining_df.shape[0]} total rows.")
    
    for keyword in keywords:
        match_mask = remaining_df[column].str.contains(keyword, case=False, na=False)
        filtered_df = remaining_df[match_mask]
        print(f"Filtering for '{keyword}': found {filtered_df.shape[0]} matching rows.")
        
        filtered_dfs[keyword] = filtered_df
        counts.append({'type': keyword, 'count': filtered_df.shape[0]})

        # Remove matched rows
        remaining_df = remaining_df[~match_mask]
        print(f"After filtering for '{keyword}', remaining rows: {remaining_df.shape[0]}")
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

    if save_after_filter and output_dir:
        filtered_path_file = os.path.join(filtered_path, f"animal_studies_metadata_after_stype_filter_{len(remaining_df)}.csv")
        remaining_df.to_csv(filtered_path_file, index=False)
        print(f"- Remaining (filtered) rows saved to '{filtered_path_file}'")
        pmid_path = os.path.join(filtered_path, f"animal_studies_metadata_after_stype_filter_{len(remaining_df)}_PMIDs.csv")
        remaining_df[["PMID"]].to_csv(pmid_path, index=False)
        print(f"- Only PMIDs from remaining rows saved to '{pmid_path}'")
        print(f"Unique PMIDs in remaining rows: {remaining_df['PMID'].nunique()}")


    return counts_df, filtered_dfs, remaining_df

def filter_and_save_disease_articles(
    remaining_df,
    filter_column='unique_conditions_linkbert_predictions',
    pattern=None,
    suffix='ms'
):
    assert pattern is not None, "A regex pattern must be provided."

    save_dir = "03_IE_ner/data/animal_studies_with_drug_disease/disease_filtered"
    os.makedirs(save_dir, exist_ok=True)

    # Filter based on pattern
    match_mask = remaining_df[filter_column].str.contains(pattern, na=False, flags=re.IGNORECASE, regex=True)
    filtered_df = remaining_df[match_mask]

    # Filenames
    base_filename = f"filtered_df_non_empty_{suffix}_{len(filtered_df)}"
    full_output_path = os.path.join(save_dir, f"{base_filename}.csv")
    pmid_output_path = os.path.join(save_dir, f"{base_filename}_PMIDs.csv")

    # Save full filtered data
    cols_to_save = ["PMID", filter_column, "unique_interventions_linkbert_predictions"]
    existing_cols = [col for col in cols_to_save if col in filtered_df.columns]
    filtered_df[existing_cols].to_csv(full_output_path, index=False)

    # Save PMIDs only
    if "PMID" in filtered_df.columns:
        filtered_df[["PMID"]].to_csv(pmid_output_path, index=False)

    print(f"- Filtered articles with '{suffix}' saved to: {full_output_path}")
    print(f"- PMIDs saved to: {pmid_output_path}")

    return filtered_df

def filter_for_diseases(remaining_df):
    diseases_to_filter = {
        "ms": r"\b(?:multiple sclerosis|ms)\b",
        "ad": r"\b(?:alzheimer(?:'s)? disease|alzheimer(?:'s)?|ad)\b",
        "tbi": r"\b(?:traumatic brain injury|tbi)\b",
        "pd": r"\b(?:parkinson(?:'s)? disease|parkinson(?:'s)?|pd)\b",
        "stroke": r"\bstroke(s)?\b",
        "als": r"\b(?:amyotrophic lateral sclerosis|als|lou gehrig(?:'s)? disease)\b"
    }

    
    all_pmids = set()  # Use a set to keep them unique

    for disease_label, pattern in diseases_to_filter.items():
        filtered_df = filter_and_save_disease_articles(
            remaining_df,
            filter_column='unique_conditions_linkbert_predictions',
            pattern=pattern,
            suffix=disease_label
        )
        if "PMID" in filtered_df.columns:
            all_pmids.update(filtered_df["PMID"].dropna().astype(str))

    # Save all unique PMIDs together
    output_dir = "03_IE_ner/data/animal_studies_with_drug_disease/disease_filtered"
    os.makedirs(output_dir, exist_ok=True)
    all_pmids_path = os.path.join(output_dir, f"all_filtered_disease_pmids_{len(all_pmids)}.csv")
    pd.DataFrame({"PMID": sorted(all_pmids)}).to_csv(all_pmids_path, index=False)
    print(f"All unique PMIDs saved to: {all_pmids_path}")

def main():
    df = pd.read_csv("./03_IE_ner/data/animal_studies_with_drug_disease/animal_studies_metadata_595768.csv") #598647

    counts_df, filtered_dfs, remaining_df = filter_and_count_by_keywords(
        df,
        column='publication_type',
        keywords=['review', 'clinical trial', 'case report', 'randomized controlled trial'],
        output_dir='./03_IE_ner/check_study_type',
        output_prefix='animal_studies_',
        save=True,
        save_after_filter=True,
        filtered_path='./03_IE_ner/data/animal_studies_with_drug_disease/'
    )
    
    #filter_for_diseases(remaining_df)
    
if __name__ == "__main__":
    main()