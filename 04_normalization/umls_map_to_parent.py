import pandas as pd
from collections import defaultdict
import json
import os
from typing import Dict, List, Set
import argparse


def load_mappings(mrrel_path, id_to_term_map_path):
    '''
    Load UMLS MRREL mappings and ID to term mapping.
    Returns:
        cui2_to_cui1: Dict mapping CUI2 to list of CUI1s
        id_to_term_map: Dict mapping CUI to term string
        parent_counts: Dict mapping parent CUI to number of children CUIs'''
    
    mappings = pd.read_csv(mrrel_path)
    mappings["cui1"] = mappings["cui1"].astype(str).str.strip()
    mappings["cui2"] = mappings["cui2"].astype(str).str.strip()
    
    print(f"Loaded {len(mappings)} relationship mappings from {mrrel_path}.")
    #print distribution over rela column
    count_rela = mappings['rela'].value_counts()
    print("Rela distribution:")
    print(count_rela)
    
    cui2_to_cui1 = defaultdict(list)

    # build mapping
    for _, row in mappings.iterrows():
        cui1 = row['cui1'].strip()
        cui2 = row['cui2'].strip()
        if pd.notna(cui1) and pd.notna(cui2):
            cui2_to_cui1[cui2].append(cui1)

    # optionally convert to a normal dict
    cui2_to_cui1 = dict(cui2_to_cui1)

    id_to_term_map = {}
    with open(id_to_term_map_path, "r", encoding="utf-8") as f:
        id_to_term_map = json.load(f)
    print(f"Loaded {len(id_to_term_map)} ID to term mappings from {id_to_term_map_path}.")
    
    parent_to_children = defaultdict(set)

    for cui2, parents in cui2_to_cui1.items():
        for cui1 in parents:
            if cui1 in id_to_term_map and cui2 in id_to_term_map:
                parent_to_children[cui1].add(cui2) 
    parent_counts = {parent: len(children) for parent, children in parent_to_children.items()}
    print(f"Computed parent counts for {len(parent_counts)} parent CUIs.")
    
    # top 5 parents with most children
    top_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 parents with most children:")
    for parent, count in top_parents:
        term = id_to_term_map.get(parent, "N/A")
        print(f"{parent} ({term}): {count} children")

        
    return cui2_to_cui1, id_to_term_map, parent_counts

def get_all_mappings_with_labels(start_cui, cui2_to_cui1, cui_to_str):
    """
    Return all CUIs reachable from `start_cui` (any depth) and their labels.

    Parameters
    ----------
    start_cui : str
        The root CUI to traverse from (will be included in `all_ids`).
    cui2_to_cui1 : dict[str, list[str]]
        Mapping dictionary where each CUI may map to one or more CUIs.
    cui_to_str : dict[str, str]
        Dictionary mapping CUI IDs to human-readable string labels.

    Returns
    -------
    all_ids : set[str]
        Set of all reachable CUIs, including `start_cui`.
    all_pairs : list[tuple[str, str]]
        List of (CUI, label) pairs for all reachable CUIs,
        **excluding the starting CUI**.
    """
    visited = set()
    stack = [start_cui]

    while stack:
        cui = stack.pop()
        if cui in visited:
            continue
        visited.add(cui)

        if cui in cui2_to_cui1:
            for nxt in cui2_to_cui1[cui]:
                if nxt not in visited:
                    stack.append(nxt)

    # Build list of (cui, label), excluding the start CUI
    all_pairs = [
        (cui, cui_to_str[cui])
        for cui in visited
        if cui != start_cui and cui in cui_to_str
    ]

    return all_pairs

def assign_nearest_dataset_parents(
    df: pd.DataFrame,
    cui2_to_cui1: Dict[str, List[str]],
    cui_to_str: Dict[str, str],
    all_umls_ids: Set[str],
    parent_counts: Dict[str, int],
    id_column: str = "drug_umls_termid",
    tokens_column: str = "drug_umls_term_norm"
) -> pd.DataFrame:
    """
    Assigns the nearest dataset parents for each row in the given DataFrame based on UMLS mappings.
    This function processes a DataFrame containing UMLS term IDs and their corresponding tokens, 
    and maps each term to its nearest parent(s) in the UMLS hierarchy. The mapping is based on 
    predefined parent-child relationships and constraints such as the maximum number of children 
    a parent can have.
    Args:
        df (pd.DataFrame): The input DataFrame containing UMLS term IDs and tokens.
        cui2_to_cui1 (Dict[str, List[str]]): A dictionary mapping child UMLS IDs (CUI2) to their 
            parent UMLS IDs (CUI1).
        cui_to_str (Dict[str, str]): A dictionary mapping UMLS IDs to their corresponding string labels.
        all_umls_ids (Set[str]): A set of valid UMLS IDs to consider for mapping.
        parent_counts (Dict[str, int]): A dictionary containing the count of children for each parent UMLS ID.
        id_column (str, optional): The column name in the DataFrame containing UMLS term IDs. 
            Defaults to "drug_umls_termid".
        tokens_column (str, optional): The column name in the DataFrame containing UMLS term tokens. 
            Defaults to "drug_umls_term_norm".
    Returns:
        pd.DataFrame: A copy of the input DataFrame with two additional columns:
            - "nearest_dataset_parent_umls": The UMLS IDs of the nearest parents for each row.
            - "nearest_dataset_parent_umls_label": The labels of the nearest parents for each row.
        defaultdict: A dictionary where keys are parent labels (with IDs) and values are sets of 
            child tokens mapped to those parents.
    Notes:
        - Rows with invalid or missing UMLS IDs are assigned "-1" as the parent ID and label.
        - Parents with more than 50 children are excluded from the mapping.
        - The function ensures alignment between UMLS IDs and tokens during processing.
    """

    parent_ids: list[str] = []
    parent_labels: list[str] = []

    mapped_to_parent = defaultdict(set)      # "drug (mapping)" -> list of children

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Raw split
        raw_ids = str(row[id_column]).split("|")
        raw_tokens = str(row[tokens_column]).split("|")
    
        # Filter based on ID != -1 while keeping alignment
        input_ids = []
        input_tokens = []
        for tid, tok in zip(raw_ids, raw_tokens):
            tid = tid.strip()
            tok = tok.strip()
            if not tid or tid == "-1":
                continue
            input_ids.append(tid)
            input_tokens.append(tok)
    
        row_parents: list[str] = []
        row_labels: list[str] = []
    
        # Loop over aligned ids + tokens
        for child_token, child_id in zip(input_tokens, input_ids):
            parents = get_all_mappings_with_labels(child_id, cui2_to_cui1, cui_to_str)  # list[(parent_id, parent_label)]
            if not parents:
                continue
    
            for parent_id, parent_label in parents:
                # Skip if parent not in counts (defensive) or too many children
                if parent_id not in parent_counts:
                    continue
                if parent_counts[parent_id] > 50:
                    continue
    
                if parent_id in all_umls_ids:
                    row_parents.append(parent_id)
                    row_labels.append(parent_label)
    
                    # Use the *token* for the child, not the ID
                    key = f"{parent_label} ({parent_id})"
                    mapped_to_parent[key].add(child_token)
                        
            
        parent_ids.append("|".join(row_parents) if row_parents else "-1")
        parent_labels.append("|".join(row_labels) if row_labels else "-1")
    
    # Attach results to a copy of the DataFrame
    result = df.copy()
    result["nearest_dataset_parent_umls"] = parent_ids
    result["nearest_dataset_parent_umls_label"] = parent_labels

    return result, mapped_to_parent

def save_stats(mapped_to_parent_clinical, mapped_to_parent_preclinical, save_folder_path="./data/umls/"):
    
    def get_df_from_mapped(mapped_to_parent):
        return pd.DataFrame([
            {
                "drug_mapping": key,
                "drug": key.split("(")[0],
                "mapping": key.split("(")[1].rstrip(")"),
                "children_values": values,
                "ner_count": len(values)
            }
            for key, values in mapped_to_parent.items()
        ])
    mapped_to_parent_df = get_df_from_mapped(mapped_to_parent_clinical)
    mapped_to_parent_df = mapped_to_parent_df.sort_values(by="ner_count", ascending=False).reset_index(drop=True)
    mapped_to_parent_df.to_csv(f'{save_folder_path}umls_mapped_to_parents_clinical_stats.csv', index=False)
    
    mapped_to_parent_preclin_df = get_df_from_mapped(mapped_to_parent_preclinical)
    mapped_to_parent_preclin_df = mapped_to_parent_preclin_df.sort_values(by="ner_count", ascending=False).reset_index(drop=True)
    mapped_to_parent_preclin_df.to_csv(f'{save_folder_path}umls_mapped_to_parents_preclinical_stats.csv', index=False)
    
def get_all_umls_ids_from_df(df: pd.DataFrame, id_column: str = "drug_umls_termid") -> Set[str]:
    all_umls_ids = {
        tid
        for cell in df[id_column]
        for tid in cell.split('|')
        if tid and tid != '-1'
    }
    return all_umls_ids

def merge_original_and_parent(
    df: pd.DataFrame,
    id_col: str = "disease_mondo_termid",
    label_col: str = "disease_term_mondo_norm",
    parent_id_col: str = "nearest_dataset_parent_mondo",
    parent_label_col: str = "nearest_dataset_parent_label",
    merged_id_col: str = "merged_mondo_termid",
    merged_label_col: str = "merged_mondo_label"
) -> pd.DataFrame:
    """
    For each row:
      1. Split `id_col` and `label_col` into parallel lists orig_ids, orig_labels.
      2. Split parent columns into parent_ids, parent_labels.
      3. Keep orig_ids & orig_labels exactly (including '-1' slots).
      4. Append any parent_id != '-1' that isn’t already in orig_ids, 
         along with its matching parent_label.
      5. Re-join into pipe-delimited merged_id_col / merged_label_col.

    Returns the modified DataFrame, and prints how many new labels were added.
    """

    df = df.copy()
    merged_ids = []
    merged_labels = []

    added_count = 0   # count new parent labels added

    for _, row in df.iterrows():
        # 1) Original
        orig_ids    = row[id_col].split("|")
        orig_labels = row[label_col].split("|")

        # 2) Parents
        parent_ids    = row[parent_id_col].split("|")
        parent_labels = row[parent_label_col].split("|")

        # 3) Start with originals in order
        mids  = list(orig_ids)
        mlabs = list(orig_labels)

        # 4) Append parents if new (keeps order)
        for pid, plab in zip(parent_ids, parent_labels):
            if pid != "-1" and pid not in mids:
                mids.append(pid)
                mlabs.append(plab)
                added_count += 1

        # ---- ORDER-PRESERVING DEDUPLICATION ----
        seen = set()
        final_ids = []
        final_labels = []

        for mid, mlab in zip(mids, mlabs):
            if mid not in seen:
                seen.add(mid)
                final_ids.append(mid)
                final_labels.append(mlab)

        # 5) Join back
        merged_ids.append("|".join(final_ids))
        merged_labels.append("|".join(final_labels))

    df[merged_id_col]    = merged_ids
    df[merged_label_col] = merged_labels

    print(f"Total new parent labels added: {added_count}")

    return df

 
def main(args):

    mrrel_path = args.mrrel_path
    id_to_term_map_path = args.id_to_term_map_path
    
    df_clinical = pd.read_csv(args.clinical_input, dtype=str)
    df_preclinical = pd.read_csv(args.preclinical_input, dtype=str)

    preclinical_output_path = args.preclinical_output
    clinical_output_path = args.clinical_output

    all_umls_ids_clinical = get_all_umls_ids_from_df(df_clinical, id_column="drug_umls_termid")
    all_umls_ids_preclinical = get_all_umls_ids_from_df(df_preclinical, id_column="drug_umls_termid")
    all_umls_ids = all_umls_ids_clinical.union(all_umls_ids_preclinical)
    
    cui2_to_cui1, cui_to_str, parent_counts = load_mappings(mrrel_path, id_to_term_map_path)
    
    df_expanded_clinical, mapped_to_parent_clinical = assign_nearest_dataset_parents(
        df_clinical,
        cui2_to_cui1,
        cui_to_str,
        all_umls_ids,
        parent_counts,
        id_column="drug_umls_termid"
    )
    df_expanded_preclinical, mapped_to_parent_preclinical = assign_nearest_dataset_parents(
        df_preclinical,
        cui2_to_cui1,
        cui_to_str,
        all_umls_ids,
        parent_counts,
        id_column="drug_umls_termid"
    )
    
    save_stats(
        mapped_to_parent_clinical,
        mapped_to_parent_preclinical,
        save_folder_path=args.stats_folder
    )
    
    df_final_preclinical = merge_original_and_parent(
        df_expanded_preclinical,
        id_col="drug_umls_termid",
        label_col="drug_umls_term_norm",
        parent_id_col="nearest_dataset_parent_umls",
        parent_label_col="nearest_dataset_parent_umls_label",
        merged_id_col="merged_umls_termid",
        merged_label_col="merged_umls_label"
    )
    
    df_final_clinical = merge_original_and_parent(
        df_expanded_clinical,
        id_col="drug_umls_termid",
        label_col="drug_umls_term_norm",
        parent_id_col="nearest_dataset_parent_umls",
        parent_label_col="nearest_dataset_parent_umls_label",
        merged_id_col="merged_umls_termid",
        merged_label_col="merged_umls_label"
    )

    df_final_preclinical.to_csv(preclinical_output_path, index=False)
    df_final_clinical.to_csv(clinical_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign UMLS parents to clinical and preclinical drug data."
    )

    # UMLS-related inputs
    parser.add_argument(
        "--mrrel_path",
        default="./data/umls/mrrel_all_drug_rela_20251209.csv",
        help="Path to mrrel CSV file."
    )
    parser.add_argument(
        "--id_to_term_map_path",
        default="./data/umls/umls_id_to_term_map.json",
        help="Path to UMLS id-to-term map JSON file."
    )

    # Input clinical & preclinical CSVs
    parser.add_argument(
        "--clinical_input",
        default="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_data_with_mondo_parents_mondo_cleaned.csv",
        help="Path to clinical input CSV."
    )
    parser.add_argument(
        "--preclinical_input",
        default="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_with_mondo_parents_mondo_cleaned.csv",
        help="Path to preclinical input CSV."
    )

    # Output paths
    parser.add_argument(
        "--preclinical_output",
        default="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_drug_data_with_umls_parents.csv",
        help="Output path for preclinical CSV."
    )
    parser.add_argument(
        "--clinical_output",
        default="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_drug_data_with_umls_parents.csv",
        help="Output path for clinical CSV."
    )

    # Where save_stats writes
    parser.add_argument(
        "--stats_folder",
        default="./data/umls/",
        help="Folder where stats from save_stats() will be written."
    )

    args = parser.parse_args()
    main(args)
