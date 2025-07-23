import pandas as pd
import re
from regex_classifiers.species_classifier import SpeciesClassifier
import os
import argparse
import glob

def clean_prediction_value(raw_value: str) -> str:
    """
    Cleans a single prediction value by removing common noise tokens.

    This includes:
    - Removing leading "n = ", "N=", "= " or similar prefixes.
    - Trimming whitespace and lowercasing the value.

    Args:
        raw_value (str): The raw prediction value from the input CSV.

    Returns:
        str: A cleaned numeric string (e.g., '172' from 'n = 172').
    """
    cleaned = raw_value.strip().lower()
    cleaned = re.sub(r"^(n\s*=\s*|=\s*)", "", cleaned)
    return cleaned


def match_doc_level_predictions(row, species_clf, context_window=50):
    """
    For each predicted number in a document, match its position in the text and determine
    whether it is used in an animal-related context based on nearby words.

    Args:
        row (pd.Series): A row containing 'prediction_encoded_label' and 'Text'.
        species_clf (SpeciesClassifier): A species classifier for contextual filtering.
        context_window (int): Number of characters before and after to consider as context.

    Returns:
        list[dict]: A list of extracted spans and classifications per value.
    """
    prediction_str = row['prediction_encoded_label']
    text = row.get('Text', '').lower()

    if not isinstance(text, str) or pd.isna(prediction_str):
        return []

    values = [v.strip() for v in str(prediction_str).split(',') if v.strip()]
    results = []

    for val in values:
        val_clean = clean_prediction_value(val)

        # Match cleaned prediction value in text
        pattern = rf'\b{re.escape(val_clean)}\b'
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            context = text[max(0, start - context_window):min(len(text), end + context_window)]

            # Classify context
            _, found_labels = species_clf.classify(context)
            is_animal_context = 1
            if len(found_labels) == 1 and found_labels[0] == "species-other":
                if not re.search(r'\banimals?\b', context, flags=re.IGNORECASE):
                    is_animal_context = 0
                else:
                    found_labels = ["animal_keyword"]

            results.append({
                "prediction_encoded_label_new": val_clean,
                "start": start,
                "end": end,
                "context": context,
                "species_classification": found_labels,
                "is_animal_context": is_animal_context,
            })

    return results

def process_predictions(ner_df, text_df, output_dir):
    print(f"Processing {len(ner_df)} NER predictions with {len(text_df)} full-text entries...")

    merged_df = ner_df.merge(text_df[['PMID', 'Text']], on='PMID', how='left')

    species_clf = SpeciesClassifier()
    print("Matching and classifying predictions...")
    merged_df['extracted_spans'] = merged_df.apply(match_doc_level_predictions, axis=1, species_clf=species_clf)

    # Flatten
    exploded_df = merged_df.explode('extracted_spans').reset_index(drop=True)
    extracted_cols = pd.json_normalize(exploded_df['extracted_spans']).reset_index(drop=True)
    exploded_df = pd.concat([exploded_df.drop(columns=['extracted_spans']), extracted_cols], axis=1)

    # Output 1: Full predictions with context
    output_with_context = os.path.join(output_dir, "animals_nr_predictions_with_context.csv")
    columns_to_keep = ["PMID", "prediction_encoded_label_new", "context", "species_classification", "is_animal_context"]
    final = exploded_df[columns_to_keep].drop_duplicates(subset=["PMID", "prediction_encoded_label_new", "context"])
    final.to_csv(output_with_context, index=False)
    print(f"Saved context-rich predictions to: {output_with_context}")

    # Filter animal-relevant only
    final = final[final['is_animal_context'] == 1]

    # Output 2: Clean document-level predictions
    grouped_df = (
        final.groupby("PMID")["prediction_encoded_label_new"]
        .apply(lambda x: ', '.join(sorted(set(x))))
        .reset_index()
        .rename(columns={"prediction_encoded_label_new": "prediction_encoded_label"})
    )

    output_clean = os.path.join(output_dir, "doc_animals_nr_predictions_clean.csv")
    grouped_df.to_csv(output_clean, index=False)
    print(f"Saved cleaned doc-level predictions to: {output_clean}")

def main():
    parser = argparse.ArgumentParser(description="Classify species relevance in NER predictions.")
    parser.add_argument(
        "--ner_file",
        default="08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions.csv",
        help="Path to a single NER predictions CSV."
    )
    parser.add_argument(
        "--ner_dir",
        help="Directory with chunked NER predictions (e.g., *_chunk_*.csv). If set, overrides --ner_file."
    )
    parser.add_argument(
        "--text_file",
        default="07_full_text_retrieval/materials_methods/combined/combined_methods.jsonl",
        help="JSONL file with PMID and full text."
    )
    parser.add_argument(
        "--output_dir",
        default="08_IE_full_text/model_predictions/animals_nr",
        help="Directory to save outputs."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load NER predictions
    if args.ner_dir:
        chunk_files = sorted(glob.glob(os.path.join(args.ner_dir, "chunk_*.csv")))
        if not chunk_files:
            raise FileNotFoundError(f"No chunked files found in {args.ner_dir}")
        print(f"Merging {len(chunk_files)} chunked prediction files...")
        ner_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    else:
        print(f"Reading NER file: {args.ner_file}")
        ner_df = pd.read_csv(args.ner_file)

    # Load text
    print(f"Reading text file: {args.text_file}")
    if args.text_file.endswith(".jsonl"):
        text_df = pd.read_json(args.text_file, lines=True)
    else:
        text_df = pd.read_csv(args.text_file)

    # Process
    process_predictions(ner_df, text_df, args.output_dir)


if __name__ == "__main__":
    main()