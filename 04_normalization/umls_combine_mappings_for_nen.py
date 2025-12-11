import os
import numpy as np
import json
import pandas as pd

def load_numpy_batches(directory_path: str, batch_name_prefix: str):
    # List batch files
    files = [
        f for f in os.listdir(directory_path)
        if f.startswith(f"{batch_name_prefix}_batch_") and f.endswith(".npy") and ("COMBINED" not in f)
    ]
    files.sort()

    # Load batches
    arrays = [np.load(os.path.join(directory_path, f)) for f in files]

    # Concatenate
    return np.concatenate(arrays, axis=0)

def load_mappings_and_embeddings(mappings_json, embeddings_directory, batch_name_prefix):
    with open(mappings_json, "r") as f:
        term_id_pairs = json.load(f)
    print(f"Loaded {len(term_id_pairs)} term-id pairs from {mappings_json}.")

    all_reps_emb = load_numpy_batches(embeddings_directory, batch_name_prefix)
    print(f"Loaded embeddings shape: {all_reps_emb.shape}")

    return term_id_pairs, all_reps_emb
    
def generate_id_to_canonical_mapping(canonical_maps_path, output_path):
    canonical_maps = pd.read_csv(canonical_maps_path)
    canonical_maps = dict(zip(canonical_maps["cui"], canonical_maps["str"]))

    print(f"Generated mapping for {len(canonical_maps)} UMLS IDs.")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(canonical_maps, f, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    batch_name_prefix = "UMLS_emb"

    print("Loading UMLS mappings and embeddings...")
    umls_mappings_json = "./data/umls/umls_term_id_pairs.json"
    umls_embeddings_directory = "./data/umls/embeddings"
    term_id_pairs, all_reps_emb_full = load_mappings_and_embeddings(umls_mappings_json, umls_embeddings_directory, batch_name_prefix) 
    
    print("Loading UMLS synonyms mappings and embeddings...")
    umls_synonyms_mappings_json = "./data/umls/umls_term_id_pairs_synonyms.json"
    umls_synonyms_embeddings_directory = "./data/umls/embeddings_synonyms"
    term_id_pairs_umls_synonyms, all_reps_emb_full_synonyms = load_mappings_and_embeddings(umls_synonyms_mappings_json, umls_synonyms_embeddings_directory, batch_name_prefix)
    
    print("Loading DrugBank extended UMLS mappings and embeddings...")
    drugbank_mappings_json = "./data/umls/umls_term_id_pairs_drugbank_ids.json"
    drugbank_embeddings_directory = "./data/umls/embeddings_drugbank_ids"
    term_id_pairs_db_ext_ids, all_reps_emb_full_db_ext = load_mappings_and_embeddings(drugbank_mappings_json, drugbank_embeddings_directory, batch_name_prefix)
    
    
    # combine all
    all_reps_emb_full_all = np.concatenate(
    [all_reps_emb_full, all_reps_emb_full_synonyms, all_reps_emb_full_db_ext],
    axis=0
    )
    term_id_pairs_all = term_id_pairs + term_id_pairs_umls_synonyms + term_id_pairs_db_ext_ids
    print(f"Combined embeddings shape: {all_reps_emb_full_all.shape}")
    print(f"Combined term-id pairs count: {len(term_id_pairs_all)}")
    
    # --- Save term-id pairs to JSON ---
    with open("./data/umls/umls_term_id_pairs_combined.json", "w", encoding="utf-8") as f:
        json.dump(term_id_pairs_all, f, ensure_ascii=False, indent=2)
        
    np.save("./data/umls/embeddings/UMLS_emb_batch_COMBINED.npy", all_reps_emb_full_all)
    
    # --- Generate id to canonical mapping ---
    canonical_maps_path = "./data/umls/mrconso_filtered_db_and_sty_474316_drug_chemical_level_0_9.csv"
    output_json_path = "./data/umls/umls_id_to_term_map.json"
    generate_id_to_canonical_mapping(canonical_maps_path, output_json_path)

