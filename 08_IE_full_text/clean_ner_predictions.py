import pandas as pd
import ast
import re
from typing import Set


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


def process_ner_entities(
    input_file: str, ner_column: str, output_file: str = None
) -> pd.DataFrame:
    """
    Read a CSV, extract and normalize unique NER entities from a specified column,
    and return a DataFrame with only ['PMID', 'Source', 'prediction_encoded_label'].

    Parameters:
        input_file (str): Path to the CSV containing the raw NER predictions.
        ner_column (str): Column name in the CSV that holds the raw entity tuples.
        output_file (str, optional): If provided, saves the processed DataFrame to this path.

    Returns:
        pd.DataFrame: Columns ['PMID', 'Source', 'prediction_encoded_label'] where
                      'prediction_encoded_label' is the cleaned, comma-separated entities.
    """
    df = pd.read_csv(input_file)
    if ner_column not in df.columns:
        raise ValueError(f"NER column '{ner_column}' not found in {input_file!r}.")

    # Extract and normalize unique entities for each row
    df["prediction_encoded_label"] = df[ner_column].apply(extract_unique_entities)

    # Keep only the needed columns
    result_df = df[["PMID", "Source", "prediction_encoded_label"]]

    if output_file:
        result_df.to_csv(output_file, index=False)

    return result_df


if __name__ == "__main__":
    # Process strains predictions
    strain_input = (
        "08_IE_full_text/model_predictions/strain/"
        "test_annotated_BioLinkBERT-base_tuples_20250430MS_part_1.csv"
    )
    strain_output = (
        "08_IE_full_text/model_predictions/strain/strain_predictions_MS.csv"
    )
    result_df = process_ner_entities(
        input_file=strain_input,
        ner_column="ner_prediction_BioLinkBERT-base_normalized",
        output_file=strain_output,
    )
    print(f"Strain entities written to: {strain_output}")

    # Process animal number predictions
    animal_input = (
        "08_IE_full_text/model_predictions/animals_nr/"
        "test_annotated_BioLinkBERT-base_tuples_20250430MS_part_1.csv"
    )
    animal_output = (
        "08_IE_full_text/model_predictions/animals_nr/"
        "animals_nr_predictions_MS.csv"
    )
    result_df = process_ner_entities(
        input_file=animal_input,
        ner_column="ner_prediction_BioLinkBERT-base_normalized",
        output_file=animal_output,
    )
    print(f"Animal entities written to: {animal_output}")
