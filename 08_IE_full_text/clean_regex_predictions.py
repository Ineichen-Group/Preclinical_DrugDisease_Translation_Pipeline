import pandas as pd
from collections import Counter
import ast
from collections import defaultdict
import argparse
import os

def document_level_strict_zero_fallback(df):
    """
    Convert sentence-level predictions to document-level.
    Keep '0' label only if no other label is present.
    """
    records = []
    for pmid, group in df.groupby('PMID'):
        # Split into non-zero and zero predictions
        nonzero = group[group['prediction_encoded_num'] > 0]
        if not nonzero.empty:
            selected = nonzero
        else:
            selected = group[group['prediction_encoded_num'] == 0]

        # Use max prediction from selected
        max_num = selected['prediction_encoded_num'].max()
        label = selected.iloc[0]['prediction_encoded_label']
        support_ids = selected['sentence_id'].tolist()

        records.append({
            'PMID': pmid,
            'prediction_encoded_num': max_num,
            'prediction_encoded_label': label,
            'supporting_sent_id': ','.join(map(str, support_ids))
        })

    return pd.DataFrame(records)

def process_species_exclude_singletons_pivoted(df):
    """
    Process species predictions at the document level with refined logic:
    - Retain species that appear in multiple sentences.
    - Allow species that appear in only one sentence *if they are the only species* (excluding species-other).
    - Remove species-other if any other species exists.
    - Include supporting sentence IDs and frequencies per species.
    """
    records = []

    for pmid, group in df.groupby('PMID'):
        species_sent_map = {}

        for _, row in group.iterrows():
            try:
                labels = ast.literal_eval(row['prediction_encoded_label'])
            except Exception:
                continue

            sent_id = row['sentence_id']
            for label in labels:
                species_sent_map.setdefault(label, set()).add(sent_id)

        # Separate out species-other
        species_other_sent_ids = species_sent_map.pop('species-other', set())

        # Identify species with more than one sentence
        multi_sent_species = {
            species: sent_ids
            for species, sent_ids in species_sent_map.items()
            if len(sent_ids) > 1
        }

        # Handle singleton species
        singleton_species = {
            species: sent_ids
            for species, sent_ids in species_sent_map.items()
            if len(sent_ids) == 1
        }

        # Include singleton species only if they are the *only* species mentioned
        if not multi_sent_species and singleton_species:
            valid_species = singleton_species
        else:
            valid_species = multi_sent_species

        # If no species left and species-other was found, restore it
        if not valid_species and species_other_sent_ids:
            valid_species = {'species-other': species_other_sent_ids}

        # If valid species include anything besides species-other, remove species-other
        if 'species-other' in valid_species and len(valid_species) > 1:
            del valid_species['species-other']

        row_data = {'PMID': pmid}
        for species, sent_ids in valid_species.items():
            row_data[f"supporting_frequency_{species}"] = len(sent_ids)
            row_data[f"supporting_sent_id_{species}"] = ','.join(map(str, sorted(sent_ids)))

        row_data['prediction_encoded_label'] = ', '.join(sorted(valid_species.keys()))
        records.append(row_data)

    return pd.DataFrame(records)

def process_assay_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process sentence-level assay predictions into document-level summaries per PMID.

    Returns a DataFrame with:
    - PMID
    - prediction_encoded_label: concatenated unique labels
    - supporting_sent_id_<category>: sentence IDs where each category appears
    - prediction_tokens_dict: full label-to-token dictionary per PMID
    """
    # Convert string representations of lists/dicts to actual Python objects
    df["prediction_encoded_label"] = df["prediction_encoded_label"].apply(ast.literal_eval)
    df["prediction_tokens"] = df["prediction_tokens"].apply(ast.literal_eval)

    # Collect all unique categories across the dataset
    all_categories = sorted({
        label for labels in df["prediction_encoded_label"] for label in labels
    })

    def aggregate_doc(group):
        pmid = group.name
        
        # Unique, sorted list of all predicted labels
        all_labels = sorted(set(label for labels in group["prediction_encoded_label"] for label in labels))
        pred_label_str = ", ".join(all_labels)

        # Create sentence_id lists for each category
        sent_id_cols = {}
        for cat in all_categories:
            sent_ids = group.loc[
                group["prediction_encoded_label"].apply(lambda labels: cat in labels),
                "sentence_id"
            ]
            sent_id_cols[f"supporting_sent_id_{cat}"] = ", ".join(map(str, sent_ids.tolist()))

        # Merge prediction_tokens dicts
        combined_tokens = defaultdict(list)
        def split_tokens(val):
            if isinstance(val, list):
                items = val
            else:
                items = [val]
            # Split on semicolons and flatten
            return [item.strip() for v in items for item in str(v).split(";")]

        for d in group["prediction_tokens"]:
            for k, v in d.items():
                combined_tokens[k].extend(split_tokens(v))

                
        def dedupe_and_join(tokens: list[str]) -> str:
            unique = sorted({ t.strip().lower() for t in tokens })
            return "; ".join(t.title() for t in unique)
        
        # Combine token matches into a readable string
        combined_tokens = {
            k: dedupe_and_join(v)
            for k, v in combined_tokens.items()
        }
     
            
        return pd.Series({
            "PMID": pmid,
            "prediction_encoded_label": pred_label_str,
            **sent_id_cols,
            "prediction_tokens_dict": combined_tokens
        })

    # Apply aggregation per PMID
    result = (
        df.groupby("PMID", group_keys=False)
        .apply(aggregate_doc)
        .reset_index(drop=True)
    )
    return result



def main():
    parser = argparse.ArgumentParser(description="Process document-level regex predictions.")
    parser.add_argument(
        "--input_dir",
        default="./model_predictions/regex",
        help="Directory containing prediction CSVs (default: './model_predictions/regex')"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    print(f" Using input directory: {input_dir}")

    # === CASES OF BINARY or HIERARCHICAL prediction ===
    prediction_files = ['blinding_predictions.csv',
                        'randomization_predictions.csv',
                        'welfare_predictions.csv',
                        'sex_predictions.csv']

    for filename in prediction_files:
        file_path = os.path.join(input_dir, filename)
        if not file_path.endswith('.csv'):
            raise ValueError(f"Input file '{file_path}' must be a CSV.")
        prediction_type = filename.split('_')[0]
        output_file = os.path.join(input_dir, f"{prediction_type}_doc_level_predictions.csv")
        df = pd.read_csv(file_path)
        doc_df = document_level_strict_zero_fallback(df)
        doc_df.to_csv(output_file, index=False)
        print(f" Saved: {output_file}")
        print(f" PMIDs processed: {len(doc_df['PMID'].unique())}")

    # === MULTI-LABEL CASES ===
    print("Processing species predictions...")
    species_file = os.path.join(input_dir, 'species_predictions.csv')
    if not species_file.endswith('.csv'):
        raise ValueError(f"Species input file '{species_file}' must be a CSV.")
    species_df = pd.read_csv(species_file)
    species_doc_df = process_species_exclude_singletons_pivoted(species_df)
    species_output_file = os.path.join(input_dir, 'species_doc_level_predictions.csv')
    species_doc_df.to_csv(species_output_file, index=False)
    print(f" Saved: {species_output_file}")
    print(f" PMIDs processed: {len(species_doc_df['PMID'].unique())}")

    print("Processing assay predictions...")
    assay_file = os.path.join(input_dir, 'assay_predictions.csv')
    if not assay_file.endswith('.csv'):
        raise ValueError(f"Assay input file '{assay_file}' must be a CSV.")
    assay_df = pd.read_csv(assay_file)
    assay_doc_df = process_assay_predictions(assay_df)
    assay_output_file = os.path.join(input_dir, 'assay_doc_level_predictions.csv')
    assay_doc_df.to_csv(assay_output_file, index=False)
    print(f" Saved: {assay_output_file}")
    print(f" PMIDs processed: {len(assay_doc_df['PMID'].unique())}")


if __name__ == '__main__':
    main()
