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
    ontology = pronto.Ontology(path_to_owl)
    mondo_terms = (t for t in ontology.terms() if t.id.startswith("MONDO:"))
    term_id_pairs = []
    all_names = []
    all_ids = []
    id_to_term = {}
    skipped = 0

    for term in tqdm(mondo_terms, desc="Filtering MONDO terms"):
        if not term.name or not term.id:
            print(f"Skipping term: {term.name, term.id}")
            skipped += 1
            continue

        all_names.append(term.name)
        all_ids.append(term.id)
        term_id_pairs.append((term.name, term.id))
        id_to_term[term.id] = term  # Add mapping from ID to term object

    print(f"Loaded {len(term_id_pairs)}. Skipped {skipped} terms.")
    return term_id_pairs, all_names, all_ids, id_to_term

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
        
def load_embed_mondo(onthology_owl_path, paht_to_save_term_id_json, path_to_save_embeddings):
    term_id_pairs, all_names, all_ids, id_to_term = load_mondo_ontology(onthology_owl_path)
    with open(paht_to_save_term_id_json, "w") as f:
        json.dump(term_id_pairs, f)
        
    embed_terms_sapbert(
        all_names=all_names,
        bs=64,
        large_batch_size=100000,
        batch_name_prefix="MONDO_emb",
        save_emb_path=path_to_save_embeddings
    )
    
def load_embed_umls():
    print("TODO")
    

def main():
    onthology_owl_path = "04_normalization/data/mondo/mondo.owl"
    paht_to_save_term_id_json = "04_normalization/data/mondo/mondo_term_id_pairs.json"
    path_to_save_embeddings = "04_normalization/data/mondo/embeddings"
    
    load_embed_mondo(onthology_owl_path, paht_to_save_term_id_json, path_to_save_embeddings)
    
if __name__ == "__main__":
    main()