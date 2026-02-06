import pandas as pd
import ast
import re
from typing import Set
import os
import argparse

def normalize_entity(text: str) -> str:
    """
    Normalize spacing and casing in entity strings such as:
        'b10 . pl'      → 'B10.Pl'
        'c57bl / 6'     → 'C57Bl/6'
        'swiss - albino'→ 'Swiss-Albino'
    Steps:
        1. Strip leading/trailing whitespace.
        2. Collapse spaces around '/', '.', and '-' characters.
        3. Convert to Title Case.
    """
    text = text.strip()
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\.\s*", ".", text)
    text = re.sub(r"\s*-\s*", "-", text)
    
    text = text.replace("##ague", "sprague")  # Specific fix for 'sprague'
    text = text.replace("##", "")  # Remove any remaining '##' artifacts
    text = text.replace("spraguedawley", "Sprague-Dawley")  # Fix concatenated form
    
    return text.title()


def is_valid_entity(text: str) -> bool:
    """
    Return True if a normalized entity should be kept; False otherwise.
    Rules:
        • Not empty or just whitespace.
        • Length > 1 character.
        • Does NOT start with the literal '##'.
    """
    text = text.strip()
    return bool(text) and len(text) > 1 and not text.startswith("##")


def extract_unique_entities(pred_str: str) -> str:
    """
    Parse a prediction string (literal list of tuples) and return a
    comma-separated, alphabetically sorted set of cleaned entity texts.

    Each tuple in the list is expected to have 4 elements, with the
    fourth element being the raw entity string.

    If parsing fails or input is invalid, returns an empty string.

    Examples:
        pred_str = "[('T1', 0, 3, 'b10 . pl'), ('T2', 4, 7, 'c57bl / 6')]"
        → "B10.Pl, C57Bl/6"
    """
    if not isinstance(pred_str, str) or not pred_str.strip().startswith("["):
        return ""

    try:
        raw_entities = ast.literal_eval(pred_str)
    except Exception as e:
        print(f"Failed to parse: {pred_str[:100]}... → {e}")
        return ""

    unique_texts: Set[str] = set()
    for entity in raw_entities:
        if (
            isinstance(entity, tuple)
            and len(entity) == 4
            and isinstance(entity[3], str)
        ):
            raw_text = entity[3]
            if is_valid_entity(raw_text):
                normalized = normalize_entity(raw_text)
                unique_texts.add(normalized)

    if not unique_texts:
        return ""
    return ", ".join(sorted(unique_texts))


def process_ner_entities_from_file(input_file: str, ner_column: str) -> pd.DataFrame:
    """
    Read a CSV, extract + normalize unique NER entities from ner_column,
    then group by PMID keeping unique entities across rows.

    Returns columns: ['PMID', 'prediction_encoded_label'].
    """
    df = pd.read_csv(input_file)
    if ner_column not in df.columns:
        raise ValueError(f"NER column '{ner_column}' not found in {input_file!r}.")

    # Parse each row into a comma-separated string of unique entities
    df["prediction_encoded_label"] = df[ner_column].apply(extract_unique_entities)

    # Drop rows where we got no entities (covers [] and parse failures)
    df["prediction_encoded_label"] = df["prediction_encoded_label"].astype(str).str.strip()
    df = df[df["prediction_encoded_label"].ne("")].copy()

    if df.empty:
        return pd.DataFrame(columns=["PMID", "prediction_encoded_label"])

    # Group by PMID and merge uniques across rows
    def merge_unique(entity_strs: pd.Series) -> str:
        uniq = set()
        for s in entity_strs.dropna().astype(str):
            for x in s.split(","):
                x = x.strip()
                if x:
                    uniq.add(x)
        return ", ".join(sorted(uniq))

    result_df = (
        df.groupby("PMID", as_index=False)["prediction_encoded_label"]
          .agg(merge_unique)
    )

    return result_df


def process_directory(
    input_dir: str,
    ner_column: str,
    output_dir: str,
    num_chunks: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Match files ending with chunk_<number>.csv
    pattern = re.compile(rf"chunk_(\d+)\.csv$")

    for filename in sorted(os.listdir(input_dir)):
        match = pattern.search(filename)
        if not match:
            continue

        chunk_number = int(match.group(1))

        # Skip chunks beyond the user-specified limit
        if chunk_number > num_chunks:
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing chunk {chunk_number}: {input_path}")

        result_df = process_ner_entities_from_file(input_path, ner_column)

        if result_df is not None:
            result_df.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
        else:
            print(f"Skipping file due to errors: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Process NER prediction files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="08_IE_full_text/model_predictions/strain/",
        help="Path to the input directory containing model predictions."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="08_IE_full_text/model_predictions/strain/processed/",
        help="Path to the output directory where processed files will be saved."
    )
    parser.add_argument(
        "--ner_column",
        type=str,
        default="ner_prediction_BioLinkBERT-base_normalized",
        help="Name of the NER prediction column to process."
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        required=True,
        help="Process chunks with index ≤ this number (e.g. 20 → chunk_1 … chunk_20)."
    )

    args = parser.parse_args()

    process_directory(
        args.input_dir,
        args.ner_column,
        args.output_dir,
        args.num_chunks
    )


if __name__ == "__main__":
    main()
