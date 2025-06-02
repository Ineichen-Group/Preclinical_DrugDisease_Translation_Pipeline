#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

# to run: extract_age_sentences.py "./model_predictions/regex/age_predictions_MS.csv" "../07_full_text_retrieval/materials_methods/combined/combined_methods_sentences_MS.csv" "./model_predictions/age/regex_age_sentences.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep only sentences where prediction_encoded_num==1 and extract their text."
    )
    parser.add_argument(
        "predictions_file",
        help="Path to CSV with predictions (must have columns: PMID, sentence_id, prediction_encoded_num, …)."
    )
    parser.add_argument(
        "sentences_file",
        help="Path to CSV with sentences (must have columns: PMID, sentence_id, sent_txt, …)."
    )
    parser.add_argument(
        "output_file",
        help="Path where the filtered output CSV (PMID, sentence_id, sent_txt) will be written."
    )
    return parser.parse_args()

def main(args):
    # 1) Read predictions and filter for prediction_encoded_num == 1
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

    # 2) Read sentences and check columns
    try:
        sents = pd.read_csv(args.sentences_file)
    except Exception as e:
        print(f"Error reading sentences file: {e}", file=sys.stderr)
        sys.exit(1)

    required_sent_cols = ["PMID", "sentence_id", "sent_txt"]
    for col in required_sent_cols:
        if col not in sents.columns:
            print(f"Missing column '{col}' in sentences file.", file=sys.stderr)
            sys.exit(1)

    # 3) Merge on (PMID, sentence_id)
    merged = pd.merge(
        preds_pos,
        sents[["PMID", "sentence_id", "sent_txt"]],
        on=["PMID", "sentence_id"],
        how="inner"
    )

    # 4) Write output CSV with only PMID, sentence_id, sent_txt
    try:
        merged.to_csv(
            args.output_file,
            index=False,
            columns=["PMID", "sentence_id", "sent_txt"]
        )
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracted {len(merged)} sentences → '{args.output_file}'")

if __name__ == "__main__":
    args = parse_args()
    main(args)
