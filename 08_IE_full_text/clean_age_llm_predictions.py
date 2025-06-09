
import re
import pandas as pd
import time

def combine_age_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a DataFrame to:
    1. Extract base document ID from 'doc_id_unique'
    2. Clean 'age_prediction' values
    3. Group by base document ID and combine unique predictions
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with 'doc_id_unique' and 'age_prediction' columns.
        
    Returns:
        pd.DataFrame: Processed DataFrame with unique combined age predictions.
    """
    # Step 1: Extract base document ID (before the last underscore)
    df['doc_base'] = df['doc_id_unique'].str.rsplit('_', n=1).str[0]

    # Step 2: Clean the age_prediction text
    # Extract AGE predictions into a new column (may include NaNs)
    df['age_extracted'] = df['age_prediction'].str.extract(r'AGE:\s*(.*?)(?=\.\s|$)', expand=False)

    # Define a helper function for cleaning age strings
    def clean_age_string(x):
        if pd.isna(x):
            return None
        return ', '.join(sorted(set([part.strip() for part in x.split(',') if part.strip()])))

    # Apply the cleaning function
    df['age_prediction_clean'] = df['age_extracted'].apply(clean_age_string)

    # Step 3: Group by base document and combine unique age predictions
    combined = (
        df.groupby('doc_base')['age_prediction_clean']
        .apply(lambda x: ', '.join(sorted(set(x.dropna()))))
        .reset_index()
        .rename(columns={'age_prediction_clean': 'age_prediction', 'doc_base': 'doc_id_unique'})
    )

    # Filter out any header row or malformed entry
    predictions_combined = combined[combined['doc_id_unique'] != 'doc_id']

    return predictions_combined

def clean_not_age(val):
    if pd.isna(val):
        return val
    parts = [p.strip() for p in val.split(',')]
    
    # Case: Only 'NOT AGE'
    if parts == ['NOT AGE']:
        return 'AGE NOT SPECIFIED'
    
    # Case: Mixed values, remove 'NOT AGE'
    parts = [p for p in parts if p != 'NOT AGE']
    
    return ', '.join(parts)

def clean_prediction(text):
    """
    Cleans age prediction text and returns a comma-separated string of unique age expressions
    in the original order (no sorting).
    - Converts 'X weeks to Y weeks' → 'X-Y weeks'
    - Removes spaces around hyphens in ranges: 'X - Y' → 'X-Y'
    - Removes weight-related entries (e.g., '200-300 g', '350 grams')
    - Filters out non-specific labels when specific ones are present
    - If nothing valid remains, returns 'age not specified'
    - Preserves input order
    """
    if pd.isna(text):
        return "age not specified"

    text = text.replace('\n', ' ').replace('\t', ' ')

    # Extract AGE: segments
    matches = re.findall(r'AGE:\s*(.*?)(?=\s*(AGE:|$|###|---))', text, flags=re.IGNORECASE)

    extracted_parts = []
    for match, _ in matches:
        match = match.strip()
        match = re.sub(r'\s+and\s+', ', ', match, flags=re.IGNORECASE)
        match = re.sub(r'(\d+)\s*(weeks?|months?|days?|years?)\s+to\s+(\d+)\s*(weeks?|months?|days?|years?)',
                       lambda m: f"{m.group(1)}-{m.group(3)} {m.group(2)}", match, flags=re.IGNORECASE)
        match = re.sub(r'(\d+)\s+to\s+(\d+)', r'\1-\2', match, flags=re.IGNORECASE)
        match = re.sub(r'\s*-\s*', '-', match)  # normalize hyphens
        extracted_parts.extend([m.strip() for m in match.split(',')])

    if not matches:
        text = text.strip()
        text = re.sub(r'\s+and\s+', ', ', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s*(weeks?|months?|days?|years?)\s+to\s+(\d+)\s*(weeks?|months?|days?|years?)',
                      lambda m: f"{m.group(1)}-{m.group(3)} {m.group(2)}", text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)\s+to\s+(\d+)', r'\1-\2', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*-\s*', '-', text)
        extracted_parts = [m.strip() for m in text.split(',')]

    # Remove weight-related expressions (e.g., 300-350 g, 200 grams, 0.5 kg)
    weight_pattern = re.compile(r'\b\d*\.?\d+\s*[-–]?\s*\d*\s*(g|gram|grams|kg|kilogram|kilograms)\b', re.IGNORECASE)
    extracted_parts = [p for p in extracted_parts if not weight_pattern.search(p)]

    # Remove empty strings and normalize
    extracted_parts = [p for p in extracted_parts if p.strip()]

    # Handle non-specific age expressions
    nonspecific_pattern = re.compile(r'^(age\s*)?(not specified|unknown|unspecified)$', re.IGNORECASE)
    specific = [p for p in extracted_parts if not nonspecific_pattern.fullmatch(p)]
    nonspecific = [p for p in extracted_parts if nonspecific_pattern.fullmatch(p)]

    seen = set()
    def unique_ordered(items):
        for item in items:
            if item not in seen:
                seen.add(item)
                yield item

    if not specific and nonspecific:
        return ', '.join(unique_ordered(nonspecific))
    elif not specific:
        return "age not specified"

    return ', '.join(unique_ordered(specific))

def main():
    model_file_name = "age_unsloth_meta_llama_3.1_8b.csv"
    model_name_str = model_file_name.replace(".csv", "").replace("-", "_")
    predictions_df = pd.read_csv(f"08_IE_full_text/model_predictions/age/{model_file_name}", names=['doc_id_unique','ent_text','age_prediction'])
    predictions_combined = combine_age_predictions(predictions_df)
    predictions_combined['age_prediction'] = predictions_combined['age_prediction'].apply(clean_not_age)
    predictions_combined['age_prediction'] = predictions_combined['age_prediction'].str.replace(r'###.*', '', regex=True).str.strip()
    predictions_combined['prediction_encoded_label'] = predictions_combined['age_prediction'].apply(clean_prediction)
    save_path= f"./08_IE_full_text/model_predictions/age/{model_name_str}_doc_level_predictions.csv"
    predictions_combined.to_csv(save_path, index=False)
    print(f"Processed {len(predictions_combined)} documents with age predictions. Saved to {save_path}")
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Done in {elapsed:.2f} seconds ({int(mins)}m {secs:.2f}s).")