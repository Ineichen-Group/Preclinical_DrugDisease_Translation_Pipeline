import pandas as pd
import ast
import re

def normalize_entity(text):
    """Normalize spacing in entity strings like 'b10 . pl', 'c57bl / 6', 'swiss - albino'."""
    text = text.strip()
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'\s*\.\s*', '.', text)
    text = re.sub(r'\s*-\s*', '-', text)
    return text.title()

def is_valid_entity(text):
    """Filter rules: not single char, not empty, not starting with ##C-"""
    text = text.strip()
    return (
        len(text) > 1 and 
        not text.startswith("##C-")
    )

def extract_unique_entities(pred_str):
    """Parse prediction string and return cleaned, unique entity texts."""
    if not isinstance(pred_str, str) or not pred_str.strip().startswith('['):
        return ''
    
    try:
        entities = ast.literal_eval(pred_str)
        unique_texts = {
            normalize_entity(text)
            for entity in entities
            if isinstance(entity, tuple) and len(entity) == 4 and isinstance(entity[3], str)
            for text in [entity[3]]
            if is_valid_entity(text)
        }
        return ', '.join(sorted(unique_texts))
    except Exception as e:
        print(f"Failed to parse: {pred_str[:100]}... → {e}")
        return ''

def process_ner_entities(input_file, ner_column, output_file=None):
    """Main processing function to clean and format NER data."""
    df = pd.read_csv(input_file)
    
    if ner_column not in df.columns:
        raise ValueError(f"NER column '{ner_column}' not found in file.")

    df['prediction_encoded_label'] = df[ner_column].apply(extract_unique_entities)
    output_df = df[['PMID', 'Source', 'prediction_encoded_label']]

    if output_file:
        output_df.to_csv(output_file, index=False)
    return output_df

result_df = process_ner_entities(
    input_file="08_IE_full_text/model_predictions/strain/test_annotated_BioLinkBERT-base_tuples_20250430MS_part_1.csv",
    ner_column="ner_prediction_BioLinkBERT-base_normalized",
    output_file="08_IE_full_text/model_predictions/strain/strain_predictions_MS.csv"
)