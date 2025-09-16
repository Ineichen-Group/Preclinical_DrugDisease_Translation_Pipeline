#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
from regex_classifiers.species_classifier import SpeciesClassifier
import re
import os
import json
species_clf = SpeciesClassifier()

# to run: python extract_age_sentences.py "./model_predictions/regex/age_predictions.csv" "../07_full_text_retrieval/materials_methods/combined/combined_methods_sentences.csv" "./model_predictions/age/regex_age_sentences.csv"
def positive_int(x: str) -> int:
    v = int(x)
    if v <= 0:
        raise argparse.ArgumentTypeError("chunksize must be a positive integer")
    return v

def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only sentences where prediction_encoded_num==1 and extract their text."
    )
    parser.add_argument(
        "predictions_file",
        type=str,
        nargs="?",
        default="./model_predictions/regex/age_predictions.csv",
        help="Path to CSV with predictions (default: ./model_predictions/regex/age_predictions.csv)"
    )
    parser.add_argument(
        "sentences_file",
        type=str,
        nargs="?",
        default="../07_full_text_retrieval/materials_methods/combined/combined_methods_sentences_27781.jsonl",
        help="Path to JSONL with sentences (default: ../07_full_text_retrieval/materials_methods/combined/combined_methods_sentences_27781.jsonl)"
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default="./model_predictions/age/regex_age_sentences.csv",
        help="Path where the filtered output CSV will be written (default: ./model_predictions/age/regex_age_sentences_27781.csv)"
    )
    
    parser.add_argument(
        "-c", "--chunksize",
        type=positive_int,
        default=250_000,
        help="Rows per chunk to read from the sentences JSONL (default: 250000). "
             "Increase for speed, decrease if memory is tight."
    )
    return parser.parse_args()

def filter_sentences_non_animal(row) -> bool:
    """Return True if sentence is animal-related, else False."""
    context = row['sent_txt'].lower() if pd.notna(row['sent_txt']) else ""
    _, found_labels = species_clf.classify(context)

    # If it's only 'species-other', check for explicit 'animal(s)' keyword
    if len(found_labels) == 1 and found_labels[0] == "species-other":
        if not re.search(r'\banimals?\b', context, flags=re.IGNORECASE):
            return False  # Not animal-related
    return True  # Keep the row

def _normalize_pmid(s: pd.Series) -> pd.Series:
    """Coerce PMID to pandas 'string' dtype, trim, and strip accidental '.0'."""
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype("string")
    else:
        s = s.astype("string")
    return s.str.strip()

def main(args):
    # --- load predictions (small) ---
    try:
        preds = pd.read_csv(
            args.predictions_file,
            dtype={"PMID": "string", "prediction_encoded_num": "Int8"},
            low_memory=False,
        )
    except Exception as e:
        print(f"Error reading predictions file: {e}", file=sys.stderr)
        sys.exit(1)

    # validate predictions
    required_pred_cols = ["PMID", "sentence_id", "prediction_encoded_num"]
    missing = [c for c in required_pred_cols if c not in preds.columns]
    if missing:
        print(f"Missing columns in predictions file: {missing}", file=sys.stderr)
        sys.exit(1)

    # keep only positives and normalize key dtypes
    preds_pos = (
        preds.loc[preds["prediction_encoded_num"] == 1, ["PMID", "sentence_id"]]
        .dropna(subset=["PMID", "sentence_id"])
        .copy()
    )
    preds_pos["PMID"] = _normalize_pmid(preds_pos["PMID"])
    # sentence_id should be integer-like; coerce safely
    if not pd.api.types.is_integer_dtype(preds_pos["sentence_id"]):
        preds_pos["sentence_id"] = pd.to_numeric(preds_pos["sentence_id"], errors="coerce").astype("Int64")
    preds_pos = preds_pos.drop_duplicates()

    # --- stream sentences JSONL in chunks & inner-join to preds_pos ---
    chunksize = getattr(args, "chunksize", 250_000)  # tune if needed
    out_cols = ["PMID", "sentence_id", "sent_txt"]
    total_found = 0
    wrote_header = False

    # ensure output dir exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    try:
        reader = pd.read_json(args.sentences_file, lines=True, chunksize=chunksize)
    except Exception as e:
        print(f"Error opening sentences file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        for i, sents_chunk in enumerate(reader, start=1):
            # validate required cols are present in this chunk
            required_sent_cols = ["PMID", "sentence_id", "sent_txt"]
            missing_chunk = [c for c in required_sent_cols if c not in sents_chunk.columns]
            if missing_chunk:
                print(f"Missing columns in sentences chunk {i}: {missing_chunk}", file=sys.stderr)
                sys.exit(1)

            # keep only needed columns & normalize dtypes to match preds_pos
            sents_chunk = sents_chunk[required_sent_cols].copy()
            sents_chunk["PMID"] = _normalize_pmid(sents_chunk["PMID"])
            if not pd.api.types.is_integer_dtype(sents_chunk["sentence_id"]):
                sents_chunk["sentence_id"] = pd.to_numeric(sents_chunk["sentence_id"], errors="coerce").astype("Int64")

            # inner join to keep only rows that appear in preds_pos
            matched = preds_pos.merge(sents_chunk, on=["PMID", "sentence_id"], how="inner")
            if not matched.empty:
                matched.to_csv(
                    args.output_file,
                    index=False,
                    columns=out_cols,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                )
                wrote_header = True
                total_found += len(matched)
    except Exception as e:
        print(f"Error while streaming/merging sentences: {e}", file=sys.stderr)
        sys.exit(1)

    # ensure the file exists with headers even if no matches
    if not wrote_header:
        pd.DataFrame(columns=out_cols).to_csv(args.output_file, index=False)

    print(f"Found {total_found} sentences with age-related predictions.")
    print(f"Extracted {total_found} sentences → '{args.output_file}'")


if __name__ == "__main__":
    args = parse_args()
    main(args)
