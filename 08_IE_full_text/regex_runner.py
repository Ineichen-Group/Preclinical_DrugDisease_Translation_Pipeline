# runner.py
"""
runner.py

Description:
    This script applies one or more regex-based classifiers (sample_size, sex,
    species, welfare, blinding, randomization, age, assay) to a CSV or JSONL file
    of texts. Each classifier processes the specified text column and writes a new
    CSV with encoded predictions and labels. You can run a single category, or run
    all categories in sequence. Large input files are streamed in chunks to keep
    memory usage manageable.

Usage:
    # 1) Classify only the "sex" category on a CSV:
    python runner.py \
        --df_path path/to/input.csv \
        --category sex \
        --text_col Text \
        --output_dir predictions/

    # 2) Run all classifiers on a JSONL in one go:
    python runner.py \
        --df_path path/to/input.jsonl \
        --category all \
        --text_col Text \
        --output_dir predictions/ \
        --chunksize 50000 \
        --progress

Arguments:
    --df_path     Path to the input file (.csv or .jsonl).
    --category    Which classifier to run: one of
                  {sample_size, sex, species, welfare, blinding,
                   randomization, age, assay, all}.
    --text_col    Name of the column containing the text to classify
                  (default: "Text").
    --output_dir  Directory to save classifier outputs (default: ./predictions).
    --chunksize   Number of rows to process per chunk (default: 50,000).
    --progress    Show a progress bar (requires tqdm).
    --resume      Append to existing outputs instead of overwriting.

Output:
    For each category run, a CSV named "<category>_predictions.csv" will be
    created in the output directory. Each CSV contains:
        - PMID (if present in the input)
        - sentence_id (if present in the input)
        - prediction_encoded_num
        - prediction_encoded_label
        - prediction_tokens (only for the assay classifier)

"""
import argparse
import os
import sys
import time
from typing import Dict, Tuple, Optional

import pandas as pd

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from regex_classifiers.sex_classifier import SexClassifier
from regex_classifiers.species_classifier import SpeciesClassifier
from regex_classifiers.welfare_classifier import WelfareClassifier
from regex_classifiers.blinding_classifier import BlindingClassifier
from regex_classifiers.randomization_classifier import RandomizationClassifier
from regex_classifiers.age_classifier import AgeClassifier
from regex_classifiers.assay_classifier import AssayClassifier
from regex_classifiers.sample_size_classifier import SampleSizeCalcClassifier

from utils.format_utils import format_species_result
from utils.format_utils import format_assay_result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run regex-based classification on a dataset of texts, streaming in chunks.\n\n"
            "Supported input formats:\n"
            "  - CSV file (e.g., .csv)\n"
            "  - JSON Lines file (e.g., .jsonl with one record per line)\n\n"
            "Supported classification categories:\n"
            "  - sample_size, sex, species, welfare, blinding, randomization, age, assay, all"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--df_path",
        required=True,
        help="Path to the input file (.csv or .jsonl).",
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=["sample_size", "sex", "species", "welfare", "blinding", "randomization", "age", "assay", "all"],
        help="Classifier to run, or 'all'.",
    )

    parser.add_argument(
        "--text_col",
        default="Text",
        help="Name of the column containing input text (default: 'Text').",
    )

    parser.add_argument(
        "--output_dir",
        default="predictions",
        help="Directory to save the classifier outputs (default: ./predictions).",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Rows per chunk to load into memory (default: 50,000).",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar (requires tqdm).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output CSVs instead of overwriting headers.",
    )

    return parser.parse_args()


# Mapping: category → (ClassifierClass, format_fn)
CLASSIFIERS = {
    "sample_size": (SampleSizeCalcClassifier, None),
    "sex": (SexClassifier, None),
    "species": (SpeciesClassifier, lambda out: format_species_result(out)),
    "welfare": (WelfareClassifier, None),
    "blinding": (BlindingClassifier, None),
    "randomization": (RandomizationClassifier, None),
    "age": (AgeClassifier, None),
    "assay": (AssayClassifier, lambda out: format_assay_result(out)),
}


def normalize_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = (
        df[text_col].astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def build_subset_columns(df_cols, category_name: str) -> Tuple[list, bool, bool, bool]:
    """Return (subset_cols, has_pmid, has_sentence_id, is_assay)."""
    has_pmid = "PMID" in df_cols
    has_sentence_id = "sentence_id" in df_cols
    is_assay = (category_name == "assay")

    subset_cols = []
    if has_pmid:
        subset_cols.append("PMID")
    if has_pmid and has_sentence_id:
        subset_cols.append("sentence_id")

    subset_cols += ["prediction_encoded_num", "prediction_encoded_label"]
    if is_assay:
        subset_cols.append("prediction_tokens")

    return subset_cols, has_pmid, has_sentence_id, is_assay


def run_classifier_on_chunk(
    df_chunk: pd.DataFrame,
    text_col: str,
    category_name: str,
    clf_obj,
    format_fn: Optional[callable],
) -> pd.DataFrame:
    """Run a single classifier on df_chunk[text_col], returning a DataFrame with new columns."""
    # Work on a view to avoid copying large frames unnecessarily
    def apply_and_format(txt):
        raw_out = clf_obj.classify(txt)
        return format_fn(raw_out) if format_fn else raw_out

    if category_name == "assay":
        df_chunk[["prediction_encoded_num", "prediction_encoded_label", "prediction_tokens"]] = (
            df_chunk[text_col].apply(lambda txt: pd.Series(apply_and_format(txt)))
        )
    else:
        df_chunk[["prediction_encoded_num", "prediction_encoded_label"]] = (
            df_chunk[text_col].apply(lambda txt: pd.Series(apply_and_format(txt)))
        )
    return df_chunk


def process_stream(
    args,
    categories: Dict[str, Tuple[type, Optional[callable]]],
):
    """Stream the input file in chunks and write intermediate CSVs per category."""
    # Sanity checks
    if not os.path.isfile(args.df_path):
        print(f"ERROR: Input file {args.df_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Reader factory
    is_jsonl = args.df_path.endswith(".jsonl")
    if is_jsonl:
        reader = pd.read_json(args.df_path, lines=True, chunksize=args.chunksize)
        print(f"Streaming JSONL input in chunks of {args.chunksize} rows...")
    else:
        reader = pd.read_csv(args.df_path, chunksize=args.chunksize)
        print(f"Streaming CSV input in chunks of {args.chunksize} rows...")

    # Instantiate classifiers once (reuse across chunks)
    active_categories = categories.keys() if args.category == "all" else [args.category]
    instantiated: Dict[str, object] = {}
    for name in active_categories:
        cls, _fmt = categories[name]
        instantiated[name] = cls()

    # Track whether we've written the header for each category
    header_written: Dict[str, bool] = {}
    for name in active_categories:
        out_path = os.path.join(args.output_dir, f"{name}_predictions.csv")
        header_written[name] = bool(args.resume and os.path.exists(out_path))

    # Progress wrapper
    if args.progress and _HAS_TQDM:
        iterator = tqdm(reader, desc="Chunks", unit="chunk")
    else:
        iterator = reader

    # Stream
    total_rows_processed = 0
    for i, df_chunk in enumerate(iterator, start=1):
        if args.text_col not in df_chunk.columns:
            print(
                f"ERROR: Specified text_col '{args.text_col}' not found in input columns for chunk {i}: "
                f"{df_chunk.columns.tolist()}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Normalize text
        df_chunk = normalize_text_column(df_chunk, args.text_col)

        # For each category, classify and append results
        for name in active_categories:
            clf_obj = instantiated[name]
            _, fmt = categories[name]

            # Run classifier on this chunk
            out_chunk = run_classifier_on_chunk(df_chunk.copy(), args.text_col, name, clf_obj, fmt)

            # Decide columns to write
            subset_cols, has_pmid, has_sentence_id, is_assay = build_subset_columns(out_chunk.columns, name)

            # Warn once per category if PMID missing
            if not has_pmid and not header_written.get(f"__warned_{name}", False):
                print(
                    f"WARNING: No 'PMID' column in input (category '{name}'). Output will only contain predictions.",
                    file=sys.stderr,
                )
                header_written[f"__warned_{name}"] = True

            # Write / append
            out_path = os.path.join(args.output_dir, f"{name}_predictions.csv")
            mode = "a" if header_written[name] else "w"
            out_chunk.to_csv(
                out_path,
                index=False,
                columns=subset_cols,
                mode=mode,
                header=not header_written[name],
            )
            header_written[name] = True

        total_rows_processed += len(df_chunk)

    print(f"Processed {total_rows_processed} rows in total.")


def main():
    args = parse_args()

    start_time = time.time()

    # Validate category
    if args.category != "all" and args.category not in CLASSIFIERS:
        print(f"ERROR: Unknown category '{args.category}'", file=sys.stderr)
        sys.exit(1)

    # If --progress requested but tqdm missing
    if args.progress and not _HAS_TQDM:
        print("NOTE: --progress requested but tqdm is not installed; continuing without a progress bar.", file=sys.stderr)

    # Kick off streaming process
    if args.category == "all":
        categories = CLASSIFIERS
    else:
        categories = {args.category: CLASSIFIERS[args.category]}

    process_stream(args, categories)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Done in {elapsed:.2f} seconds ({int(mins)}m {secs:.2f}s).")


if __name__ == "__main__":
    main()