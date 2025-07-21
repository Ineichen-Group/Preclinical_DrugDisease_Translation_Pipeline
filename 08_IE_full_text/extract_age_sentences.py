#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
from regex_classifiers.species_classifier import SpeciesClassifier
import re
import json
species_clf = SpeciesClassifier()

# to run: python extract_age_sentences.py "./model_predictions/regex/age_predictions.csv" "../07_full_text_retrieval/materials_methods/combined/combined_methods_sentences.csv" "./model_predictions/age/regex_age_sentences.csv"

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

def main(args):
    try:
        preds = pd.read_csv(args.predictions_file)
    except Exception as e:
        print(f"Error reading predictions file: {e}", file=sys.stderr)
        sys.exit(1)

    required_pred_cols = ["PMID", "sentence_id", "prediction_encoded_num"]
    for col in required_pred_cols:
        if col not in preds.columns:
            print(f"Missing column '{col}' in predictions file.", file=sys.stderr)
            sys.exit(1)

    preds_pos = preds.loc[
        preds["prediction_encoded_num"] == 1,
        ["PMID", "sentence_id"]
    ]

    try:
        sents = pd.read_json(args.sentences_file, lines=True)
    except Exception as e:
        print(f"Error reading sentences file: {e}", file=sys.stderr)
        sys.exit(1)

    required_sent_cols = ["PMID", "sentence_id", "sent_txt"]
    for col in required_sent_cols:
        if col not in sents.columns:
            print(f"Missing column '{col}' in sentences file.", file=sys.stderr)
            sys.exit(1)

    merged = pd.merge(
        preds_pos,
        sents[["PMID", "sentence_id", "sent_txt"]],
        on=["PMID", "sentence_id"],
        how="inner"
    )

    print(f"Found {len(merged)} sentences with age-related predictions.")           
    # Filter rows that are not animal-related
    #merged = merged[merged.apply(filter_sentences_non_animal, axis=1)]

    try:
        merged.to_csv(
            args.output_file,
            index=False,
            columns=["PMID", "sentence_id", "sent_txt"]
        )
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(merged)} animal-related sentences → '{args.output_file}'")


if __name__ == "__main__":
    args = parse_args()
    main(args)
