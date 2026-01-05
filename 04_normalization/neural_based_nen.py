import os
os.environ["WANDB_DISABLED"] = "true"  # set BEFORE importing transformers

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
#from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Any, Tuple, Dict, List, Union
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import time
import argparse

def load_embeddings(directory_path_embeddings, batch_name_prefix, directory_path_terms_ids_json):
    print(f"Loading embeddings from {directory_path_embeddings} with prefix {batch_name_prefix}...")
    print(f"Loading term-id pairs from {directory_path_terms_ids_json}...")
    
    # List all npy files in the directory
    files = [f for f in os.listdir(directory_path_embeddings) if f.endswith(".npy")]

    # 1) If a COMBINED file exists → load ONLY that one
    combined_files = [f for f in files if f.endswith("COMBINED.npy")]
    
    if combined_files:
        print("Found COMBINED embeddings file, loading that one...")
        combined_files.sort()
        emb_path = os.path.join(directory_path_embeddings, combined_files[0])
        all_reps_emb_full = np.load(emb_path)
    else:
        print("No COMBINED embeddings file found, loading all batch files...")
        # 2) Otherwise load all batch files
        batch_files = sorted(
            f for f in files
            if f.startswith(f"{batch_name_prefix}_batch_") and f.endswith(".npy")
        )
        if not batch_files:
            raise FileNotFoundError("No embedding files found.")
        all_reps_emb_full = np.concatenate(
            [np.load(os.path.join(directory_path_embeddings, f)) for f in batch_files],
            axis=0
        )

    # Load term-id pairs
    with open(directory_path_terms_ids_json, "r") as f:
        term_id_pairs = json.load(f)

    # Validate length
    if len(all_reps_emb_full) != len(term_id_pairs):
        raise ValueError(
            f"Embeddings={len(all_reps_emb_full)} but term_id_pairs={len(term_id_pairs)}"
        )

    return all_reps_emb_full, term_id_pairs


    
def map_query_to_terminology(query: str, 
                        tokenizer: AutoTokenizer, 
                        model: AutoModel, 
                        all_reps_emb_full: torch.Tensor, 
                        ontology_sf_id_pairs: np.ndarray, 
                        canonical_mapping_dict: Dict[str, str] = None,
                        dist_threshold: float = 15,  # Added threshold parameter
                        n_entities: int = 5, 
                        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        ) -> Tuple[int, str, str, List[Tuple[str, int]], float]:
    
    """
    Map a query to the closest ontology concept using a pre-trained model and return its canonical form.
    If the distance to the nearest concept exceeds the threshold, return the original query.

    Parameters:
    - query (str): The input query string to be mapped.
    - tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the query.
    - model (PreTrainedModel): The pre-trained model used to generate embeddings for the query.
    - all_reps_emb_full (torch.Tensor): The array of embeddings for all ontology concepts.
    - ontology_sf_id_pairs (np.ndarray): The array of ontology concept ID and label pairs.
    - canonical_mapping_dict (Dict[str, str]): A dictionary mapping ontology IDs to their canonical forms.
    - dist_threshold (float): Distance threshold for deciding whether to map the query.
    - n_entities (int): The number of nearest entities to retrieve.

    Returns:
    - Tuple[int, str, str, List[Tuple[str, int]], float]: The predicted ontology concept ID or 0 for no mapping,
      label, its canonical form or original query, a list of the nearest entities, and the minimum distance.
    """
        
   # assume model & all_reps_emb_full are already on `device`
    query_toks = tokenizer(
        [query],
        padding="max_length",
        max_length=25,
        truncation=True,
        return_tensors="pt",
    )
    query_toks = {k: v.to(device) for k, v in query_toks.items()}

    with torch.inference_mode():
        out = model(**query_toks)
    query_cls_rep = out[0][:, 0, :]  # [CLS], shape (1, D)

    # distances on the SAME device
    dist = torch.cdist(query_cls_rep, all_reps_emb_full)  # (1, N)
    nn_index = torch.argmin(dist).item()
    min_distance = dist[0, nn_index].item()

    # nearest n
    nearest_n_indices = torch.argsort(dist[0])[:n_entities]
    nearest_n_entities = [ontology_sf_id_pairs[idx.item()] for idx in nearest_n_indices]

    if min_distance > dist_threshold:
        return -1, query, query, nearest_n_entities, round(min_distance, 4)

    predicted_label = ontology_sf_id_pairs[nn_index]
    predicted_term = predicted_label[0]
    predicted_id = predicted_label[1]

    if canonical_mapping_dict:
        canonical_form = canonical_mapping_dict.get(predicted_id, predicted_term)
    else:
        canonical_form = predicted_term

    return predicted_id, predicted_term, canonical_form, nearest_n_entities, round(min_distance, 4)
    
def process_row_annotations(
    row: Union[str, float], 
    tokenizer: Any, 
    model: Any, 
    all_reps_emb_full: Any, 
    ontology_sf_id_pairs: Dict[str, str], 
    canonical_mapping_dict: Dict[str, str],
    dist_threshold=10, 
    n_entities=3,
    device: torch.device = None,
) -> Tuple[str, str, str, str, str, Dict[str, List[str]], Dict[str, str]]:
    """
    Processes a row of annotations, mapping terms to ontology CT concepts and returning the results.

    Parameters:
    - row (Union[str, float]): A string of terms separated by '|', or NaN.
    - tokenizer (Any): The tokenizer used for mapping terms.
    - model (Any): The model used for mapping terms.
    - all_reps_emb_full (Any): The embeddings used for mapping terms.
    - ontology_sf_id_pairs (Dict[str, str]): Dictionary of ontology ID and term pairs.
    - canonical_mapping_dict (Dict[str, str]): Dictionary mapping terms to their canonical forms.
    - dist_threshold (float): Similarity distance threshold for deciding whether to map the query.
    - n_entities (int): The number of nearest entities to retrieve.

    Returns:
    - Tuple[str, str, str, str, str, Dict[str, List[str]], Dict[str, str]]:
        - Concatenated ontology terms.
        - Concatenated ontology term IDs.
        - Concatenated canonical forms of the ontology terms.
        - Concatenated closest 3 entities.
        - Concatenated minimum distances.
        - Dictionary mapping canonical forms to lists of terms.
        - Dictionary mapping terms to their canonical forms.
    """    
    if pd.isna(row) or not isinstance(row, str):
        return "", "", "", "", "", {}, ""

    terms = row.split("|")
    ontology_terms = []
    ontology_terms_canonical = []
    ontology_termids = []
    closest_3_entites = []
    min_distances = []

    norm_to_terms: Dict[str, List[str]] = {}
    term_to_norm: Dict[str, str] = {}

    for term in terms:
        predicted_id, predicted_label, canonical_form, n_3_entities, nn_distance = map_query_to_terminology(
            term,
            tokenizer,
            model,
            all_reps_emb_full,
            ontology_sf_id_pairs,
            canonical_mapping_dict=canonical_mapping_dict,
            dist_threshold=dist_threshold,
            n_entities=n_entities,
            device=device,
        )
        ontology_terms.append(predicted_label)
        ontology_terms_canonical.append(canonical_form)
        ontology_termids.append(predicted_id)
        min_distances.append(nn_distance)
        closest_3_entites.append(n_3_entities)

        norm_to_terms.setdefault(canonical_form, []).append(term)
        term_to_norm[term] = canonical_form

    for key in norm_to_terms:
        norm_to_terms[key] = list(set(norm_to_terms[key]))

    return (
        "|".join(map(str, ontology_terms)),
        "|".join(map(str, ontology_termids)),
        "|".join(ontology_terms_canonical),
        "|".join([str(ents) for ents in closest_3_entites]),
        "|".join([str(dist) for dist in min_distances]),
        norm_to_terms,
        term_to_norm,
    )

def process_row_annotations_from_cache(
    row: Union[str, float],
    term2mapping: Dict[str, Tuple[int, str, str, List[Tuple[str, int]], float]],
) -> Tuple[str, str, str, str, str, Dict[str, List[str]], Dict[str, str]]:

    if pd.isna(row) or not isinstance(row, str):
        return "", "", "", "", "", {}, ""

    terms = row.split("|")
    ontology_terms = []
    ontology_termids = []
    ontology_terms_canonical = []
    closest_3_entites = []
    min_distances = []

    norm_to_terms: Dict[str, List[str]] = {}
    term_to_norm: Dict[str, str] = {}

    for term in terms:
        term = term.strip()
        if not term:
            continue

        # should always be present, but fallback just in case
        mapped = term2mapping.get(term, (-1, term, term, [], float("inf")))
        predicted_id, predicted_label, canonical_form, n_3_entities, nn_distance = mapped

        ontology_terms.append(predicted_label)
        ontology_termids.append(predicted_id)
        ontology_terms_canonical.append(canonical_form)
        closest_3_entites.append(n_3_entities)
        min_distances.append(nn_distance)

        norm_to_terms.setdefault(canonical_form, []).append(term)
        term_to_norm[term] = canonical_form

    for key in norm_to_terms:
        norm_to_terms[key] = list(set(norm_to_terms[key]))

    return (
        "|".join(map(str, ontology_terms)),
        "|".join(map(str, ontology_termids)),
        "|".join(ontology_terms_canonical),
        "|".join([str(ents) for ents in closest_3_entites]),
        "|".join([str(dist) for dist in min_distances]),
        norm_to_terms,
        term_to_norm,
    )



def normalize_ner_columns_no_cache(
    data_dir,
    df,
    col_to_map,
    tokenizer,
    model,
    terminology="mondo",
    dist_threshold=10,
    n_entities=3,
    device: torch.device = None,
):
    """
    Normalize named entity recognition (NER) columns in a DataFrame using a pretrained embedding model
    and a controlled vocabulary (terminology).

    Parameters:
    ----------
    data_dir : str
        Base directory where embeddings and mapping files are stored.
    df : pandas.DataFrame
        Input DataFrame containing the column with NER strings to normalize.
    col_to_map : str
        Column name in `df` containing NER entities to be normalized.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used for encoding the input strings.
    model : transformers.PreTrainedModel
        Language model used to generate embeddings for input strings.
    terminology : str, optional (default="mondo")
        Name of the controlled vocabulary used for normalization (e.g., "mondo", "hpo").
    dist_threshold : float, optional (default=10)
        Maximum cosine distance threshold for considering a match valid.
    n_entities : int, optional (default=3)
        Number of closest matches to return.

    Returns:
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
        - 'linkbert_<terminology>_<entity_type>'
        - '<terminology>_termid'
        - '<terminology>_term_norm'
        - '<terminology>_closest_3'
        - '<terminology>_cdist'

    Notes:
    -----
    Assumes that entity types end with a suffix like "_conditions".
    Mapping files must be present in the expected directory structure.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct paths to embedding and mapping files
    embedding_dir = f"{data_dir}{terminology}/embeddings"
    embedding_prefix = f"{terminology.upper()}_emb"
    term_id_path = f"{data_dir}{terminology}/{terminology}_term_id_pairs.json"

    # Load precomputed embeddings and corresponding term IDs
    all_reps_emb_full_np, term_id_pairs = load_embeddings(
        embedding_dir, embedding_prefix, term_id_path
    )
    all_reps_emb_full = torch.from_numpy(all_reps_emb_full_np).to(device)
    all_reps_emb_full.requires_grad_(False)

    model = model.to(device)
    model.eval()

    print(f"Loaded embeddings: {all_reps_emb_full.shape}, term_id_pairs: {len(term_id_pairs)}")
    tqdm.pandas(desc=f"Mapping {col_to_map} NER to {terminology}")

    entity_type = col_to_map.split("_")[-1]

    id_to_term_path = f"{data_dir}{terminology}/{terminology}_id_to_term_map.json"
    with open(id_to_term_path, "r", encoding="utf-8") as f:
        canonical_mapping_dict = json.load(f)

    df[
        [
            f"linkbert_{terminology}_{entity_type}",
            f"{terminology}_termid",
            f"{terminology}_term_norm",
            f"{terminology}_closest_3",
            f"{terminology}_cdist",
        ]
    ] = df[col_to_map].progress_apply(
        lambda x: pd.Series(
            process_row_annotations(
                x,
                tokenizer,
                model,
                all_reps_emb_full,
                term_id_pairs,
                canonical_mapping_dict,
                dist_threshold,
                n_entities,
                device=device,
            )[:5]
        )
    )

    return df

def load_relevant_embeddings_and_mappings(
    data_dir: str,
    terminology: str,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    """
    Load precomputed embeddings and term-ID pairs for a given terminology.

    Parameters:
    - data_dir (str): Base directory where embeddings and mapping files are stored.
    - terminology (str): Name of the controlled vocabulary (e.g., "mondo", "hpo").
    - device (torch.device): Device to load the embeddings onto.

    Returns:
    - Tuple[torch.Tensor, List[Tuple[str, str]]]: 
        - Tensor of embeddings on the specified device.
        - List of term-ID pairs.
        - Dictionary mapping ontology IDs to their canonical forms.
    """
    embedding_dir = f"{data_dir}{terminology}/embeddings"
    embedding_prefix = f"{terminology.upper()}_emb"
    if terminology.lower() == "umls":
        term_id_path = f"{data_dir}{terminology}/{terminology}_term_id_pairs_combined.json"
    else:
        term_id_path = f"{data_dir}{terminology}/{terminology}_term_id_pairs.json"
        
    id_to_term_path = f"{data_dir}{terminology}/{terminology}_id_to_term_map.json"
    
    all_reps_emb_full_np, term_id_pairs = load_embeddings(
        embedding_dir, embedding_prefix, term_id_path
    )
    all_reps_emb_full = torch.from_numpy(all_reps_emb_full_np).to(device)
    all_reps_emb_full.requires_grad_(False)
    with open(id_to_term_path, "r", encoding="utf-8") as f:
            canonical_mapping_dict = json.load(f)
            
    print(f"Loaded embeddings: {all_reps_emb_full.shape}, term_id_pairs: {len(term_id_pairs)}, canonical mappings: {len(canonical_mapping_dict)}")         
    return all_reps_emb_full, term_id_pairs, canonical_mapping_dict

def normalize_ner_columns(
    data_dir,
    df,
    col_to_map,
    tokenizer,
    model,
    terminology="mondo",
    dist_threshold=10,
    n_entities=3,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    
    # Load precomputed embeddings and corresponding term IDs
    all_reps_emb_full, term_id_pairs, canonical_mapping_dict = load_relevant_embeddings_and_mappings(data_dir=data_dir, terminology=terminology, device=device)
    
    tqdm.pandas(desc=f"Mapping {col_to_map} NER to {terminology}")

    entity_type = col_to_map.split("_")[-1]

    # Build cache for all unique terms
    term2mapping = build_term_mapping(
        df,
        col_to_map,
        tokenizer,
        model,
        all_reps_emb_full,
        term_id_pairs,
        canonical_mapping_dict,
        dist_threshold,
        n_entities,
        device,
    )
    
    if terminology == 'umls':
        prefix = "drug"
    else:
        prefix = "disease"
        

    # Use cache row-wise (no more model calls here)
    df[
        [
            f"linkbert_{terminology}_{entity_type}",
            f"{prefix}_{terminology}_termid",
            f"{prefix}_{terminology}_term_norm",
            f"{prefix}_{terminology}_closest_3",
            f"{prefix}_{terminology}_cdist",
        ]
    ] = df[col_to_map].progress_apply(
        lambda x: pd.Series(
            process_row_annotations_from_cache(x, term2mapping)[:5]
        )
    )

    return df


def get_unique_terms(column):
    return set(term.strip() for row in column.dropna() for term in str(row).split('|') if term.strip())

def extract_unique_terms(df: pd.DataFrame, col_to_map: str) -> List[str]:
    unique_terms = set()
    for row in df[col_to_map].dropna():
        for t in str(row).split("|"):
            t = t.strip()
            if t:
                unique_terms.add(t)
    return list(unique_terms)

def build_term_mapping(
    df: pd.DataFrame,
    col_to_map: str,
    tokenizer,
    model,
    all_reps_emb_full: torch.Tensor,
    ontology_sf_id_pairs,
    canonical_mapping_dict: Dict[str, str],
    dist_threshold: float,
    n_entities: int,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, Tuple[int, str, str, List[Tuple[str, int]], float]]:
    """
    Precompute mapping for all unique terms in df[col_to_map].
    Returns: term -> (pred_id, pred_term, canonical_form, nearest_n_entities, min_distance)
    """
    unique_terms = extract_unique_terms(df, col_to_map)
    print(f"Found {len(unique_terms)} unique terms in '{col_to_map}'")

    term2mapping: Dict[str, Tuple[int, str, str, List[Tuple[str, int]], float]] = {}

    # ensure on device
    model = model.to(device)
    all_reps_emb_full = all_reps_emb_full.to(device)

    for start in tqdm(range(0, len(unique_terms), batch_size), desc="Embedding unique terms"):
        batch_terms = unique_terms[start : start + batch_size]

        # 1) embed batch of terms
        enc = tokenizer(
            batch_terms,
            padding=True,
            truncation=True,
            max_length=25,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out = model(**enc)
        batch_embs = out[0][:, 0, :]  # CLS embeddings: (B, D)

        # 2) distance to ontology embeddings: (B, N)
        dist_matrix = torch.cdist(batch_embs, all_reps_emb_full)

        # 3) per-term nearest neighbors
        for i, term in enumerate(batch_terms):
            dists = dist_matrix[i]  # (N,)
            nn_index = torch.argmin(dists).item()
            min_distance = dists[nn_index].item()

            nearest_n_indices = torch.argsort(dists)[:n_entities]
            nearest_n_entities = [ontology_sf_id_pairs[idx.item()] for idx in nearest_n_indices]

            if min_distance > dist_threshold:
                term2mapping[term] = (-1, term, term, nearest_n_entities, round(min_distance, 4))
            else:
                predicted_label = ontology_sf_id_pairs[nn_index]
                predicted_term = predicted_label[0]
                predicted_id = predicted_label[1]

                canonical_form = canonical_mapping_dict.get(predicted_id, predicted_term) \
                    if canonical_mapping_dict else predicted_term

                term2mapping[term] = (
                    predicted_id,
                    predicted_term,
                    canonical_form,
                    nearest_n_entities,
                    round(min_distance, 4),
                )

    return term2mapping

def generate_mapping_stats(df, col_to_map, log_dir, time_taken="n.a", terminology="mondo"):
    # Count total and successfully mapped terms
    total_terms = 0
    successfully_mapped = 0
    entity_type = col_to_map.split("_")[-1]
    unmapped_rows = []
    mapped_rows = []
    
    original_col = col_to_map

    for idx, row in df.iterrows():
        mentions_raw = row.get(original_col, "")
        ids_raw = row.get(f"{terminology}_termid", "")
        candidates_raw = row.get(f"{terminology}_closest_3", "")
        distances_raw = row.get(f"{terminology}_cdist", "")
        
        if pd.isna(ids_raw):
            ids_raw = ""

        id_list = [id_.strip() for id_ in str(ids_raw).split('|') if id_.strip()]
        mention_list = [m.strip() for m in str(mentions_raw).split('|') if m.strip()]
        candidates_list = [c.strip() for c in str(candidates_raw).split('|')]
        distance_list = [d.strip() for d in str(distances_raw).split('|')]
        
        total_terms += len(id_list)
        successful = sum(1 for id_ in id_list if id_ != "-1")
        successfully_mapped += successful

        # Collect any individual failures (where id == "-1")
        for mention, id_, candidate, dist in zip(mention_list, id_list, candidates_list, distance_list):
            if id_ == "-1":
                unmapped_rows.append({
                    "mention": mention,
                    f"{terminology}_termid": id_,
                    f"{terminology}_closest_3": candidate,
                    f"{terminology}_cdist": dist,
                    f"{terminology}_cdist_list": distance_list
                })
            else:
                mapped_rows.append({
                    "mention": mention,
                    f"{terminology}_termid": id_,
                    f"{terminology}_closest_3": candidate,
                    f"{terminology}_cdist": dist,
                    f"{terminology}_cdist_list": distance_list
                })

    print(f"Total {entity_type} mentions: {total_terms}")
    print(f"Successfully mapped to {terminology} (term_id != -1): {successfully_mapped}")
    print(f"Mapping success rate: {successfully_mapped / total_terms * 100:.2f}%")

    # Get unique terms before and after mapping
    unique_before = get_unique_terms(df[original_col])
    unique_after = get_unique_terms(df[f'linkbert_mapped_{entity_type}'])
    unique_after_neural_map = get_unique_terms(df[f'linkbert_{terminology}_{entity_type}'])

    # Get counts
    count_before = len(unique_before)
    count_after = len(unique_after)
    count_after_mondo = len(unique_after_neural_map)

    print(f"Unique {entity_type} before mapping: {count_before}")
    print(f"Unique {entity_type} after mapping to dict: {count_after}")
    print(f"Unique {entity_type} after mapping to {terminology}: {count_after_mondo}")
    
    # Save all print outputs to a log dictionary
    log_data = {
        "source_col_mapped": col_to_map,
        "unique_before_any_mapping": count_before,
        "unique_after_dict": count_after,
        f"unique_after_{terminology}": count_after_mondo,
        "total_condition_mentions_for_mapping": total_terms,
        f"successfully_mapped_{terminology}": successfully_mapped,
        "mapping_success_rate_percent": round(successfully_mapped / total_terms * 100, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken_for_NEN": time_taken
    }

    # Convert to single-row DataFrame
    log_df = pd.DataFrame([log_data])

    # Append to CSV
    log_df.to_csv(log_dir + f"{entity_type}_{terminology}_mapping_stats.csv", index=False)
    
    pd.DataFrame(unmapped_rows).to_csv(log_dir + f"{entity_type}_{terminology}_failed_mapping_cases.csv", index=False)
    pd.DataFrame(mapped_rows).to_csv(log_dir + f"{entity_type}_{terminology}_success_mapping_cases.csv", index=False)

def main(
    mapping_type,
    col_to_map,
    data_dir,
    input_file,
    output_file,
    stats_dir,
    save_stats=False,
    terminology=None,
    dist_threshold=None,
):
    assert mapping_type in ["disease", "drug"], "Type must be 'disease' or 'drug'"

    print(f"Input file: {input_file}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)

    # Load data
    df = pd.read_csv(input_file)

    if "nct_id" in df.columns:
        df = df[["nct_id", col_to_map]]
    elif "PMID" in df.columns:
        df = df[["PMID", col_to_map]]
    else:
        raise ValueError(
            f"{input_file}: neither 'nct_id' nor 'PMID' column found"
        )
    n_entities = 3

    # ---- Defaults if not provided as arguments
    if terminology is None:
        if mapping_type == "disease":
            terminology = "mondo"
        else:
            terminology = "umls"

    if dist_threshold is None:
        if mapping_type == "disease":
            dist_threshold = 9.65
        else:
            dist_threshold = 8.20

    print(f"Using terminology: {terminology}")
    print(f"Using distance threshold: {dist_threshold}")

    # Normalize and time
    print(f"Starting normalization for: {mapping_type.upper()} with cdist {dist_threshold}")
    start_time = time.time()
    df_mapped = normalize_ner_columns(
        data_dir, df, col_to_map, tokenizer, model,
        terminology, dist_threshold, n_entities, device=device
    )
    elapsed = time.time() - start_time
    time_taken = str(timedelta(seconds=int(elapsed)))

    print(f"Normalization time for '{col_to_map}': {time_taken}")
    df_mapped.to_csv(output_file, index=False)
    print(f"Output saved to: {output_file}")
    
    if save_stats:
        generate_mapping_stats(
            df_mapped, col_to_map, log_dir=stats_dir,
            time_taken=time_taken, terminology=terminology
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize NER columns for diseases or drugs.")
    parser.add_argument(
        '--type',
        type=str,
        choices=["disease", "drug"],
        default="disease",  
        help="Which type to normalize (default: disease)"
    )
    parser.add_argument(
        '--col_to_map',
        type=str,
        default="linkbert_mapped_conditions",  # linkbert_mapped_drugs
        help="Column that contains the entities for normalization."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default="04_normalization/data/", 
        help="Path to a base directory where embeddings and mapping files are stored."
    )
    parser.add_argument(
        '--input',
        type=str,
        default="04_normalization/data/mapped_to_embeddings_ontologies/drug_disease_mapped_preclinical_extra_studies.csv", 
        help="Path to the input CSV file."
    ) 
    parser.add_argument(
        '--output',
        type=str,
        default="04_normalization/data/mapped_to_embeddings_ontologies/drug_disease_mapped_preclinical_extra_studies.csv", 
        help="Path to the output CSV file."
    )
    parser.add_argument(
        '--stats_dir',
        type=str,
        default="04_normalization/nen_stats/", 
        help="Path to save normalization stats like count of unique entities before and after linking."
    )
    parser.add_argument(
        '--terminology',
        type=str,
        choices=["mondo", "umls"],
        default=None,
        help="Terminology to use (e.g. 'mondo' for diseases, 'umls' for drugs). If not set, it is inferred from --type."
    )
    parser.add_argument(
        '--dist_threshold',
        type=float,
        default=None,
        help="Distance threshold for deciding whether to map a mention. If not set, a default is chosen based on --type."
    )
    
    args = parser.parse_args()
    main(
        args.type,
        args.col_to_map,
        args.data_dir,
        args.input,
        args.output,
        args.stats_dir,
        save_stats=False,
        terminology=args.terminology,
        dist_threshold=args.dist_threshold,
    )