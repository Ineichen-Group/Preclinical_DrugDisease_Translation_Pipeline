import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Any, Tuple, Dict, List, Union
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import time
import argparse

def load_embeddings(directory_path_embeddings, batch_name_prefix, directory_path_terms_ids_json):
    # List all files in the directory that match the pattern
    files = [f for f in os.listdir(directory_path_embeddings) if f.startswith(f'{batch_name_prefix}_batch_') and f.endswith('.npy')]
    # Sort files to maintain the order, especially important if the batch index is used in processing
    files.sort()

    # Initialize an empty list to hold the data from each file
    all_data = []

    # Load each file and append the data to the list
    for file in files:
        file_path = os.path.join(directory_path_embeddings, file)
        data = np.load(file_path)
        all_data.append(data)

    all_reps_emb_full = np.concatenate(all_data, axis=0)
    
    with open(directory_path_terms_ids_json, "r") as f:
        term_id_pairs = json.load(f)
        
    return all_reps_emb_full, term_id_pairs
    
def map_query_to_terminology(query: str, 
                        tokenizer: PreTrainedTokenizer, 
                        model: PreTrainedModel, 
                        all_reps_emb_full: np.ndarray, 
                        ontology_sf_id_pairs: np.ndarray, 
                        canonical_mapping_dict: Dict[str, str] = None,
                        dist_threshold: float = 15,  # Added threshold parameter
                        n_entities: int = 5) -> Tuple[int, str, str, List[Tuple[str, int]], float]:
    
    """
    Map a query to the closest ontology concept using a pre-trained model and return its canonical form.
    If the distance to the nearest concept exceeds the threshold, return the original query.

    Parameters:
    - query (str): The input query string to be mapped.
    - tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the query.
    - model (PreTrainedModel): The pre-trained model used to generate embeddings for the query.
    - all_reps_emb_full (np.ndarray): The array of embeddings for all ontology concepts.
    - ontology_sf_id_pairs (np.ndarray): The array of ontology concept ID and label pairs.
    - canonical_mapping_dict (Dict[str, str]): A dictionary mapping ontology IDs to their canonical forms.
    - dist_threshold (float): Distance threshold for deciding whether to map the query.
    - n_entities (int): The number of nearest entities to retrieve.

    Returns:
    - Tuple[int, str, str, List[Tuple[str, int]], float]: The predicted ontology concept ID or 0 for no mapping,
      label, its canonical form or original query, a list of the nearest entities, and the minimum distance.
    """
        
    # Encode the query
    query_toks = tokenizer.batch_encode_plus([query], 
                                             padding="max_length", 
                                             max_length=25, 
                                             truncation=True,
                                             return_tensors="pt")
    if torch.cuda.is_available():
        query_toks = query_toks.to('cuda')  # Move tensors to GPU
        
    # Get the model output
    with torch.no_grad():
        query_output = model(**query_toks)
    
    # Extract the CLS token representation
    query_cls_rep = query_output[0][:, 0, :]

    # Compute distances between query embedding and all concept embeddings
    if torch.cuda.is_available():
        dist = torch.cdist(query_cls_rep, all_reps_emb_full)
        nn_index = torch.argmin(dist).item()  # This finds the index of the minimum value
        min_distance = dist[0, nn_index].item()  # Extract the minimum distance at that index
    else:
        dist = cdist(query_cls_rep.cpu().detach().numpy(), all_reps_emb_full)
        nn_index = np.argmin(dist).item()
        min_distance = dist[0, nn_index]  # Since dist is a numpy array, get the minimum distance at the index 

    # Retrieve the nearest n_entities
    nearest_n_entities = []
    if torch.cuda.is_available():
        nearest_n_indices = torch.argsort(dist[0])[:n_entities]  # Get indices of the n smallest distances
    else:
        nearest_n_indices = np.argsort(dist[0])[:n_entities]
    for idx in nearest_n_indices:
        nearest_n_entities.append(ontology_sf_id_pairs[idx.item()])

    if min_distance > dist_threshold:
        # If distance is greater than the threshold, return the original query with no mapping
        return -1, query, query, nearest_n_entities, round(min_distance, 4)
        
    # Get the predicted concept ID and label
    predicted_label = ontology_sf_id_pairs[nn_index]
    predicted_term = predicted_label[0]
    predicted_id = predicted_label[1]

    # Get the canonical form from the dictionary
    if canonical_mapping_dict:
        canonical_form = canonical_mapping_dict.get(predicted_id, "Canonical form not found")
    else:
        canonical_form = predicted_term

    # Return the predicted concept ID, label, and canonical form
    return predicted_id, predicted_term, canonical_form, nearest_n_entities, round(min_distance, 4)
    
def process_row_annotations(
    row: Union[str, float], 
    tokenizer: Any, 
    model: Any, 
    all_reps_emb_full: Any, 
    ontology_sf_id_pairs: Dict[str, str], 
    canonical_mapping_dict: Dict[str, str],
    dist_threshold=10, 
    n_entities=3
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
        # Return empty strings and empty dictionaries for all the values
        return "", "", "", {}, {}, "", ""
    
    terms = row.split('|')
    ontology_terms = []
    ontology_terms_canonical = []
    ontology_termids = []
    ontology_norms = []
    closest_3_entites = []
    min_distances = []  # List to store minimum distances

    # Dictionaries to track mappings
    norm_to_terms = {}  # ontology norm as key, list of terms as values
    term_to_norm = {}   # Each term from the row and the ontology norm to which it was mapped

    for term in terms:
        predicted_id, predicted_label, canonical_form, n_3_entities, nn_distance = map_query_to_terminology(term, tokenizer, model, all_reps_emb_full, ontology_sf_id_pairs, canonical_mapping_dict=canonical_mapping_dict, dist_threshold=dist_threshold, n_entities=n_entities)
        ontology_terms.append(predicted_label)
        ontology_terms_canonical.append(canonical_form)
        ontology_termids.append(predicted_id)
        min_distances.append(nn_distance)
        closest_3_entites.append(n_3_entities)

        # Populate dictionaries
        #print(canonical_form)
        if canonical_form in norm_to_terms:
            norm_to_terms[canonical_form].append(term)
        else:
            norm_to_terms[canonical_form] = [term]

        term_to_norm[term] = canonical_form

    # Ensure unique terms in norm_to_terms dictionary
    for key in norm_to_terms:
        norm_to_terms[key] = list(set(norm_to_terms[key]))

    return '|'.join(ontology_terms), '|'.join(map(str, ontology_termids)), '|'.join(ontology_terms_canonical), '|'.join([str(ents) for ents in closest_3_entites]), '|'.join([str(dist) for dist in min_distances]),  norm_to_terms, term_to_norm


def normalize_ner_columns(
    data_dir,
    df,
    col_to_map,
    tokenizer,
    model,
    terminology="mondo",
    dist_threshold=10,
    n_entities=3
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

    # Construct paths to embedding and mapping files
    embedding_dir = f"{data_dir}{terminology}/embeddings"
    embedding_prefix = f"{terminology.upper()}_emb"
    term_id_path = f"{data_dir}{terminology}/{terminology}_term_id_pairs.json"

    # Load precomputed embeddings and corresponding term IDs
    all_reps_emb_full, term_id_pairs = load_embeddings(
        embedding_dir, embedding_prefix, term_id_path
    )

    # Move embeddings and model to GPU if available
    if torch.cuda.is_available():
        all_reps_emb_full = torch.tensor(all_reps_emb_full).to("cuda")
        model = model.to("cuda")

    print(f"Loaded embeddings: {all_reps_emb_full.shape}, term_id_pairs: {len(term_id_pairs)}")
    tqdm.pandas(desc=f"Mapping {col_to_map} NER to {terminology}")

    # Extract entity type from the column name (assumes a suffix pattern)
    entity_type = col_to_map.split("_")[-1]

    # Load ID-to-term mapping if applicable (only for certain entity types)
    if entity_type == "conditions":
        id_to_term_path = f"{data_dir}{terminology}/{terminology}_id_to_term_map.json"
        with open(id_to_term_path, "r", encoding="utf-8") as f:
            canonical_mapping_dict = json.load(f)
    else:
        canonical_mapping_dict = None

    # Normalize each row in the specified column using a helper function
    df[
        [
            f"linkbert_{terminology}_{entity_type}",
            f"{terminology}_termid",
            f"{terminology}_term_norm",
            f"{terminology}_closest_3",
            f"{terminology}_cdist"
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
                n_entities
            )[:5]
        )
    )

    return df

def get_unique_terms(column):
    return set(term.strip() for row in column.dropna() for term in str(row).split('|') if term.strip())

def generate_mapping_stats(df, col_to_map, log_dir, time_taken="n.a", terminology="mondo"):
    # Count total and successfully mapped terms
    total_terms = 0
    successfully_mapped = 0
    entity_type = col_to_map.split("_")[-1]
    unmapped_rows = []
    mapped_rows = []
    if entity_type == "drugs":
        original_col = 'unique_interventions_linkbert_predictions'
    else:
        original_col = f'unique_{entity_type}_linkbert_predictions'

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

def main(mapping_type, col_to_map, data_dir, input_file, output_file, stats_dir, save_stats=True):
    assert mapping_type in ["disease", "drug"], "Type must be 'disease' or 'drug'"

    print(f"Input file: {input_file}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # Load data
    df = pd.read_csv(input_file)
    #df = df.head(15)
    # Columns and output path

    n_entities=3
    if mapping_type == "disease":
        terminology = "mondo"
        output_file = output_file
        dist_threshold=9.7
    else:
        terminology = "umls"
        output_file = output_file 
        dist_threshold=10

    # Normalize and time
    print(f"Starting normalization for: {mapping_type.upper()} with cdist {dist_threshold}")
    start_time = time.time()
    df_mapped = normalize_ner_columns(data_dir, df, col_to_map, tokenizer, model, terminology, dist_threshold, n_entities)
    elapsed = time.time() - start_time
    time_taken = str(timedelta(seconds=int(elapsed)))

    print(f"Normalization time for '{col_to_map}': {time_taken}")

    # Stats and output
    if save_stats:
        generate_mapping_stats(df_mapped, col_to_map, log_dir=stats_dir, time_taken=time_taken, terminology=terminology)
    df_mapped.to_csv(output_file, index=False)
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize NER columns for diseases or drugs.")
    parser.add_argument(
        '--type',
        type=str,
        choices=["disease", "drug"],
        default="drug",  
        help="Which type to normalize (default: disease)"
    )
    parser.add_argument(
        '--col_to_map',
        type=str,
        default="linkbert_mapped_conditions", #linkbert_mapped_drugs
        help="Column that contains the entities for normalization."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default="04_normalization/data/ner_samples/sampled_drugs_manual_map.csv", 
        help="Path to the input CSV file (default: chunks/dict_mapped_ner_chunk_1.csv)"
    )
    parser.add_argument(
        '--input',
        type=str,
        default="04_normalization/data/ner_samples/sampled_drugs_manual_map.csv", 
        help="Path to the input CSV file (default: chunks/dict_mapped_ner_chunk_1.csv)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="04_normalization/data/mapped_to_embeddings_ontologies/sampled_drugs_manual_map_umls_pred.csv", 
        help="Path to the output CSV file (default: chunks/dict_mapped_ner_chunk_1.csv)"
    )
    parser.add_argument(
        '--stats_dir',
        type=str,
        default="04_normalization/nen_stats/", 
        help="Path to save normalization stats like count of unique entities before and after linking."
    )
    
    args = parser.parse_args()
    main(args.type, args.col_to_map, args.data_dir, args.input, args.output, args.stats_dir)