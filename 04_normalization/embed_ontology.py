import pandas as pd
import ast
import numpy as np
import os
import pronto
from itertools import islice
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

def load_mondo_ontology(path_to_owl):
    print("Loading MONDO")
    ontology = pronto.Ontology(path_to_owl)
    mondo_terms = (t for t in ontology.terms() if t.id.startswith("MONDO:"))

    term_id_pairs = []
    seen_names = set()       # for checking uniqueness
    all_names = []
    seen_name_set = set()
    all_ids = []             # IDs (can repeat)
    id_to_term_name = {}     # ID → canonical name
    skipped = 0

    for term in tqdm(mondo_terms, desc="Filtering MONDO terms"):
        if not term.name or not term.id:
            print(f"Skipping term: {term.name, term.id}")
            skipped += 1
            continue

        canonical_name = term.name.strip()
        term_id = term.id.strip()

        if canonical_name not in seen_names:
            term_id_pairs.append((canonical_name, term_id))
            seen_names.add(canonical_name)
            all_names.append(canonical_name)
            all_ids.append(term_id)

        # Add synonyms
        for syn in term.synonyms:
            synonym_name = syn.description.strip()
            if synonym_name not in seen_names:
                term_id_pairs.append((synonym_name, term_id))
                seen_names.add(synonym_name)
                all_names.append(synonym_name)
                all_ids.append(term_id)

        # Map ID to canonical name
        id_to_term_name[term_id] = canonical_name

    print(f"Loaded {len(term_id_pairs)} ({len(all_names)}, {len(all_ids)}) names (including synonyms). Skipped {skipped} terms.")
    return term_id_pairs, all_names, all_ids, id_to_term_name

def embed_batch_and_save_batch(tokenizer, model, name_subset, batch_idx, bs, save_emb_path, batch_name_prefix):
    """
    Embeds a subset of names using the given tokenizer and model, and saves the embeddings to disk.

    Parameters:
    - tokenizer: HuggingFace tokenizer
    - model: HuggingFace model
    - name_subset (List[str]): List of names to embed
    - batch_idx (int): Index of the current large batch (used in filename)
    - bs (int): Mini-batch size for tokenization
    - save_emb_path (str): Path to save the embeddings
    - batch_name_prefix (str): Prefix for the saved file name
    """
    all_reps = []

    for i in tqdm(range(0, len(name_subset), bs), desc=f"Embedding mini-batches for batch {batch_idx}"):
        mini_batch = name_subset[i:i + bs]

        # Tokenize
        toks = tokenizer.batch_encode_plus(
            mini_batch,
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt"
        )

        # Forward pass (adjust for CUDA if needed)
        # toks = {k: v.cuda() for k, v in toks.items()}  # if using GPU
        output = model(**toks)
        cls_rep = output[0][:, 0, :]  # CLS token representation
        all_reps.append(cls_rep.cpu().detach().numpy())

    # Concatenate and save
    all_reps_emb_full = np.concatenate(all_reps, axis=0)
    save_path = f"{save_emb_path}/{batch_name_prefix}_batch_{batch_idx}.npy"
    np.save(save_path, all_reps_emb_full)
    print(f"Saved batch {batch_idx} embeddings to {save_path}")


def embed_terms_sapbert(
    all_names,
    bs=128,
    large_batch_size=100000,
    batch_name_prefix="MONDO_emb",
    save_emb_path="./embeddings"
):
    """
    Processes a list of names in large batches and saves their embeddings.

    Parameters:
    - all_names (List[str]): Full list of names to embed
    - bs (int): Mini-batch size used within each large batch (default: 128)
    - large_batch_size (int): Size of large batches to chunk input list (default: 100000)
    - batch_name_prefix (str): Prefix for saved batch files (default: "MONDO_emb")
    - save_emb_path (str): Directory to save .npy embedding files (default: "./embeddings")
    """
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    # Optional: move to GPU
    # model = model.cuda()

    print(f"Embedding {len(all_names)} names in chunks of {large_batch_size}...")
    for batch_start in range(0, len(all_names), large_batch_size):
        batch_end = min(batch_start + large_batch_size, len(all_names))
        name_subset = all_names[batch_start:batch_end]
        batch_idx = batch_start // large_batch_size

        embed_batch_and_save_batch(
            tokenizer=tokenizer,
            model=model,
            name_subset=name_subset,
            batch_idx=batch_idx,
            bs=bs,
            save_emb_path=save_emb_path,
            batch_name_prefix=batch_name_prefix
        )
        
def load_embed_mondo(onthology_owl_path, path_to_save_term_id_json, path_to_save_id_to_term_json, path_to_save_embeddings):
    term_id_pairs, all_names, all_ids, id_to_term = load_mondo_ontology(onthology_owl_path)
    with open(path_to_save_term_id_json, "w") as f:
        json.dump(term_id_pairs, f)
    with open(path_to_save_id_to_term_json, "w") as f:
        json.dump(id_to_term, f)
        
    embed_terms_sapbert(
        all_names=all_names,
        bs=64,
        large_batch_size=100000,
        batch_name_prefix="MONDO_emb",
        save_emb_path=path_to_save_embeddings
    )
    
def load_umls_terms(umls_mrconso_path):
    print("Loading UMLS")
    unique_cui_umls = pd.read_csv(umls_mrconso_path)
    unique_cui_umls['str'] = unique_cui_umls['str'].str.strip()
    unique_cui_umls['cui'] = unique_cui_umls['cui'].str.strip()

    # Extract cleaned values as lists
    all_names = unique_cui_umls['str'].values.tolist()
    all_ids = unique_cui_umls['cui'].values.tolist()
    
    term_id_pairs = []
    #term_id_pairs_with_sty = []

    for term_name, term_id in zip(all_names, all_ids):
        #term_sty = cui_sty_dict[term_id]
        #term_name_sty = str(term_name) + f" ({term_sty})"
        term_id_pairs.append((term_name, term_id))
        #term_id_pairs_with_sty.append((term_name_sty, term_id))
    print(f"Loaded {len(term_id_pairs)}")
    return term_id_pairs, all_names, all_ids
    
    
def load_embed_umls(umls_mrconso_path, paht_to_save_term_id_json, path_to_save_embeddings):
    term_id_pairs, all_names, all_ids = load_umls_terms(umls_mrconso_path)
    with open(paht_to_save_term_id_json, "w") as f:
        json.dump(term_id_pairs, f)
    embed_terms_sapbert(
        all_names=all_names,
        bs=64,
        large_batch_size=100000,
        batch_name_prefix="UMLS_emb",
        save_emb_path=path_to_save_embeddings
    )
    

def main(load_mondo=False, load_umls=True):
    if load_mondo:
        onthology_owl_path = "04_normalization/data/mondo/mondo.owl"
        paht_to_save_term_id_json = "04_normalization/data/mondo/mondo_term_id_pairs.json"
        path_to_save_id_to_term_json =  "04_normalization/data/mondo/mondo_id_to_term_map.json"
        path_to_save_embeddings = "04_normalization/data/mondo/embeddings"
        
        load_embed_mondo(onthology_owl_path, paht_to_save_term_id_json, path_to_save_id_to_term_json, path_to_save_embeddings)
    if load_umls:
        umls_mrconso_path = "04_normalization/data/umls/mrconso_filtered_db_and_sty_474316_drug_chemical_level_0_9.csv"
        paht_to_save_term_id_json_umls = "04_normalization/data/umls/umls_term_id_pairs.json"
        path_to_save_embeddings_umls = "04_normalization/data/umls/embeddings"
        
        load_embed_umls(umls_mrconso_path, paht_to_save_term_id_json_umls, path_to_save_embeddings_umls)
    
    
if __name__ == "__main__":
    main(load_mondo=True, load_umls=False)