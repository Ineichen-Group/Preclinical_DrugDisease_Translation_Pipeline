# runner.py
"""
runner.py

Description:
    This script applies one or more regex‐based classifiers (sex, species, welfare,
    blinding, randomization) to a CSV file of texts. Each classifier processes the
    specified text column and outputs a new CSV containing encoded predictions and
    labels. You can run a single category or all categories in sequence.

Usage:
    # 1) Classify only the "sex" category:
    python runner.py \
        --df_path path/to/input.csv \
        --category sex \
        --text_col Text \
        --output_dir predictions/

    # 2) Run all classifiers in one go ("sex", "species", "welfare", "blinding", "randomization"):
    python runner.py \
        --df_path path/to/input.csv \
        --category all \
        --text_col Text \
        --output_dir predictions/


Output:
    For each category run, a CSV named "<category>_predictions_MS.csv" will be created
    in the output directory. Each output CSV has columns:
        - PMID (if present in the input)
        - prediction_encoded_num
        - prediction_encoded_label
Example:
    Assuming you have `input.csv` with a column named "Text" (and optional "PMID"),
    to classify all categories and save outputs under ./predictions/:

        python runner.py \
            --df_path ./input.csv \
            --category all \
            --text_col Text \
            --output_dir ./predictions/


"""
import argparse
import os
import sys
import time

import pandas as pd

from classifiers.sex_classifier import SexClassifier
from classifiers.species_classifier import SpeciesClassifier
from classifiers.welfare_classifier import WelfareClassifier
from classifiers.blinding_classifier import BlindingClassifier
from classifiers.randomization_classifier import RandomizationClassifier
from classifiers.age_classifier import AgeClassifier

from utils.format_utils import format_species_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run regex‐based classification (sex, species, welfare, blinding, randomization, or all) on a CSV of texts."
    )
    parser.add_argument(
        "--df_path",
        required=True,
        help="Path to input CSV (must contain a column 'Text' or as specified).",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=["sex", "species", "welfare", "blinding", "randomization", "age", "all"],
        help="Which classifier to run: 'sex', 'species', 'welfare', 'blinding', 'randomization', or 'all'.",
    )
    parser.add_argument(
        "--text_col",
        default="Text",
        help="Name of the column in the CSV that contains the full text to classify.",
    )
    parser.add_argument(
        "--output_dir",
        default="predictions",
        help="Directory where output CSV(s) should be written.",
    )
    return parser.parse_args()


def run_and_save(
    category_name: str,
    classifier_cls,
    format_fn,
    df: pd.DataFrame,
    text_col: str,
    output_dir: str,
):
    """
    Instantiates the given classifier_cls, applies it to every row in df[text_col],
    uses format_fn to turn classifier.classify(text) → (num, label) or (vector_str, labels_str),
    then writes a CSV named "<category_name>_predictions_MS.csv" containing:
      [PMID?, prediction_encoded_num, prediction_encoded_label].
    """

    # 1) Instantiate and apply classifier
    clf = classifier_cls()
    df_copy = df.copy()

    def apply_and_format(txt):
        raw_out = clf.classify(txt)
        # format_fn should return a 2‐element tuple (num_str, label_str)
        return format_fn(raw_out) if format_fn else raw_out

    # The result of .apply(...) must be a Series of length‐2 tuples
    df_copy[["prediction_encoded_num", "prediction_encoded_label"]] = df_copy[text_col].apply(
        lambda txt: pd.Series(apply_and_format(txt))
    )

    # 2) Build output filename & path
    out_filename = f"{category_name}_predictions_MS.csv"
    out_path = os.path.join(output_dir, out_filename)

    # 3) Decide which columns to write: include "PMID" if present
    if "PMID" in df_copy.columns:
        if "sentence_id" in df_copy.columns:
            subset_cols = ["PMID", "sentence_id", "prediction_encoded_num", "prediction_encoded_label"]
        else:
            subset_cols = ["PMID", "prediction_encoded_num", "prediction_encoded_label"]
    else:
        subset_cols = ["prediction_encoded_num", "prediction_encoded_label"]
        print(
            f"WARNING: No 'PMID' column in input (category '{category_name}'). "
            "Output will only contain predictions.",
            file=sys.stderr,
        )

    # 4) Write out CSV (no index)
    df_copy[subset_cols].to_csv(out_path, index=False)
    print(f"{category_name.capitalize()} classification completed → {out_path}")


def main():
    args = parse_args()

    # 1) Read input CSV, check existence
    if not os.path.isfile(args.df_path):
        print(f"ERROR: Input file {args.df_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.df_path)
    if args.text_col not in df.columns:
        print(
            f"ERROR: Specified text_col '{args.text_col}' not found in CSV columns: {df.columns.tolist()}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2) Normalize whitespace in the text column (optional but recommended)
    df[args.text_col] = (
        df[args.text_col].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 3) Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 4) Define our mapping: category → (ClassifierClass, format_fn)
    # format_fn = None means classifier.classify(...) already returns (num, label)
    CLASSIFIERS = {
        "sex": (SexClassifier, None),
        "species": (SpeciesClassifier, format_species_result),
        "welfare": (WelfareClassifier, None),
        "blinding": (BlindingClassifier, None),
        "randomization": (RandomizationClassifier, None),
        "age": (AgeClassifier, None),
    }

    if args.category != "all":
        # Run just the requested classifier
        if args.category not in CLASSIFIERS:
            print(f"ERROR: Unknown category '{args.category}'", file=sys.stderr)
            sys.exit(1)

        cls, fmt = CLASSIFIERS[args.category]
        run_and_save(args.category, cls, fmt, df, args.text_col, args.output_dir)

    else:
        # Run all categories in sequence
        for name, (cls, fmt) in CLASSIFIERS.items():
            run_and_save(name, cls, fmt, df, args.text_col, args.output_dir)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Done in {elapsed:.2f} seconds ({int(mins)}m {secs:.2f}s).")
