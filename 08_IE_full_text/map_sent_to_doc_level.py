import pandas as pd
from collections import Counter
import ast

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

def process_species_exclude_singletons(df):
    """
    Process species predictions at the document level with updated logic:
    - Exclude species mentioned only once in one sentence.
    - Include 'species-other' only if no other species remain.
    - Include supporting sentence IDs.
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

        # Filter: only keep species that occur in more than one sentence
        filtered_species = {
            species: sent_ids
            for species, sent_ids in species_sent_map.items()
            if len(sent_ids) > 1
        }

        # If only species-other is present, keep it
        if not filtered_species and 'species-other' in species_sent_map:
            filtered_species = {
                'species-other': species_sent_map['species-other']
            }

        # If other species exist, remove species-other
        elif 'species-other' in filtered_species and len(filtered_species) > 1:
            del filtered_species['species-other']

        for species, sent_ids in filtered_species.items():
            records.append({
                'PMID': pmid,
                'species': species,
                'supporting_frequency': len(sent_ids),
                'supporting_sent_id': ','.join(map(str, sorted(sent_ids)))
            })

    return pd.DataFrame(records)


def main():
        # === CONFIGURATION ===
    INPUT_FILES = ['08_IE_full_text/model_predictions/regex/blinding_predictions_MS.csv',
                  '08_IE_full_text/model_predictions/regex/randomization_predictions_MS.csv',
                  '08_IE_full_text/model_predictions/regex/welfare_predictions_MS.csv',
                  '08_IE_full_text/model_predictions/regex/sex_predictions_MS.csv']
    
    for file in INPUT_FILES:
        if not file.endswith('.csv'):
            raise ValueError(f"Input file '{file}' must be a CSV.")
        prediction_type = file.split('/')[-1].split('_')[0]
        OUTPUT_FILE = f'08_IE_full_text/model_predictions/regex/{prediction_type}_doc_level_predictions_MS.csv'
        df = pd.read_csv(file)
        doc_df = document_level_strict_zero_fallback(df)
        doc_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Document-level predictions saved to '{OUTPUT_FILE}'")
        print(f"Processed and saved PMIDs: {len(doc_df['PMID'].unique())}")
        
    # Process species predictions with refined logic
    species_file = '08_IE_full_text/model_predictions/regex/species_predictions_MS.csv'
    if not species_file.endswith('.csv'):
        raise ValueError(f"Species input file '{species_file}' must be a CSV.")
    species_df = pd.read_csv(species_file)
    species_doc_df = process_species_exclude_singletons(species_df)
    species_output_file = '08_IE_full_text/model_predictions/regex/species_doc_level_predictions_MS.csv'
    species_doc_df.to_csv(species_output_file, index=False)
    print(f"Species document-level predictions saved to '{species_output_file}'")
    print(f"Processed and saved PMIDs: {len(doc_df['PMID'].unique())}")

if __name__ == '__main__':
    main()
