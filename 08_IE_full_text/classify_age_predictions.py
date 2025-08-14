import pandas as pd
import ast
import re
import argparse
from pathlib import Path

def classify_age(age_in_weeks_str):
    """
    Classify age based on the input string.
    :param age_in_weeks_str: String representing age in weeks.
    :return: Age classification as 'young', 'adult', or 'aged'.
    """

    try:
        age_in_weeks = float(age_in_weeks_str)
    except (ValueError, TypeError):
        print(f"Invalid age input: {age_in_weeks_str}")
        return None  

    if age_in_weeks < 8:
        return "young"
    elif 8 <= age_in_weeks < 60:
        return "adult"
    else:
        return "aged"
    
def map_to_weeks(pred_str, time_base):
    if time_base == "days" or time_base == "day" or time_base == "d":
        # Convert days to weeks
        try:
            days = int(pred_str)
            return str(days // 7)  # Convert days to weeks
        except ValueError:
            return "unknown"
    elif time_base == "months" or time_base == "month" or time_base == "mo":
        # Convert months to weeks (approx. 4.345 weeks per month)
        try:
            months = int(pred_str)
            return str(int(months * 4.345))  # Convert months to weeks
        except ValueError:
            return "unknown"
    elif time_base == "years" or time_base == "year":  
        # Convert years to weeks (52 weeks per year)
        try:
            years = int(pred_str)
            return str(years * 52)  # Convert years to weeks
        except ValueError:
            return "unknown"
    else:
        return "unknown"  # If time base is not recognized


def normalize_age_string(age: str) -> str:
    if not isinstance(age, str):
        return age

    age = age.replace("<", "").replace(">", "")
    age = age.replace("years old", "years")
    age = age.replace("year old", "years")
    age = age.replace("days old", "days")
    age = age.replace("day old", "days")
    # Convert things like "10-week-old" or "1011-week-old" to "10 weeks"
    age = re.sub(r'(\d+)-week-old', r'\1 weeks', age)

    # Normalize dashes and spaces
    age = age.replace('–', '-').replace('—', '-').replace('~', '-')
    age = re.sub(r'\s*-\s*', '-', age)

    return age

def process_age_predictions(pmid, pred_str):
    """
    Process age predictions from a DataFrame column.
    
    :param pred_str: String representation of age predictions.
    :return: Age classification as 'young', 'adult', or 'aged'.
    """
    if not isinstance(pred_str, str):
        return "unknown"
    print(f"Processing PMID: {pmid}")
    pred_str = pred_str.strip().lower()
    #print(pred_str)
    if pred_str == "adult":
        return "adult"
    if pred_str == "age not specified":
        return "age not specified"
    
    # Handle cases with multiple predictions
    predictions = pred_str.split(",")
    print(f"Processing predictions: {predictions}")
    
    classifications = []
    normalized_age = []
    for pred in predictions:
        pred = pred.strip().lower()
        if pred == "adult":
            classifications.append("adult")
            continue
        if pred == "age not specified":
            continue
        if pred == "juvenile":
            classifications.append("young")
            continue
        try:
            if len(pred.split(" ")) != 2:
                pred = normalize_age_string(pred)
            if len(pred.split(" ")) != 2:
                print(f"Skipping malformed prediction: {pred}")
                continue
            age, time_base = pred.split(" ")
        except ValueError:
            print(f"Skipping malformed prediction: {pred}")
            continue
        
        #age = age.replace("–", "-").replace("~", "-")  # Normalize dash characters
        age = normalize_age_string(age)
        if age.count('.') > 1 or re.search(r"[^\d.\s\-]", age):
            print(f"Skipping malformed prediction with multiple dots or invalid characters: {pred}")
            continue
        if time_base == "weeks":
            if "-" not in age:
               # Handle cases of wrongly formatted ages 
               if float(age) > 150:
                   if len(age) == 3:
                       age = age[0] + "-" + age[1:] # assume a dash is missing and the first digit is part of the range
                   else:
                       age = age[0:2] + "-" + age[2:] # assume a dash is missing and the first two digits are part of the range
            normalized_age.append(age + " weeks")             
            age_range_values = age.split("-")
            for age_value in age_range_values:
                age_value = age_value.strip()
                age_classification = classify_age(age_value)
                classifications.append(age_classification)
                
        else:
            if (time_base == "days") and ("-" not in age):
               # Handle cases of wrongly formatted ages 
               if float(age) > 1000:
                    age = age[0:2] + "-" + age[2:] # assume a dash is missing and the first two digits are part of the range
            age_range_values = age.split("-")
            for age_value in age_range_values:
                age_value = age_value.strip()
                age_in_weeks = map_to_weeks(age_value, time_base)
                if age_in_weeks != "unknown":
                    age_classification = classify_age(age_in_weeks)
                    classifications.append(age_classification)
                    normalized_age.append(age_in_weeks + " weeks")
    mapped_classifications = ", ".join(sorted({c for c in classifications if c is not None}))
    mapped_normalized_age = ", ".join(sorted({c for c in normalized_age if c is not None}))
    print(f"Mapped classifications: {mapped_classifications}")
    return mapped_classifications, mapped_normalized_age
    
    
def classify_age_prediction(prediction_df, pred_col_species="prediction_encoded_label", pred_col_age="verified_prediction_encoded_label", species_df=None):
    """
    Classify age predictions based on sentence-level species annotations.
    Only classify if supporting sentences contain only 'mouse', 'rat', or species-other.
    If any non-allowed species (other than species-other) is present, prediction is skipped.
    
    :param prediction_df: DataFrame with age predictions
    :param pred_col_species: Column with prediction strings for species
    :param species_df: Sentence-level species annotations
    :return: DataFrame with age classifications
    """
    prediction_df = prediction_df.copy()

    if species_df is not None:
        species_df = species_df.copy()

        # Parse list-like strings to real lists
        species_df[pred_col_species] = species_df[pred_col_species].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

        # Add sentence-level unique ID
        species_df['sentence_uid'] = species_df['PMID'].astype(str) + '_' + species_df['sentence_id'].astype(str)

        # Build map: sentence_uid → species label list
        sentence_species_map = dict(zip(species_df['sentence_uid'], species_df[pred_col_species]))

        def is_valid_species(supporting_ids_str):
            if pd.isna(supporting_ids_str):
                return False
            sentence_ids = re.split(r'[;,]', supporting_ids_str)
            sentence_ids = [sid.strip() for sid in sentence_ids]

            all_species = set()
            for sid in sentence_ids:
                labels = sentence_species_map.get(sid, [])
                # Exclude "species-other"
                filtered = [label for label in labels if label != 'species-other']
                all_species.update(filtered)

            # Allow only if remaining species are in {mouse, rat}
            return all_species.issubset({'mouse', 'rat'})

        # Flag valid rows
        prediction_df['is_mouse_or_rat_only'] = prediction_df['supporting_sentence_ids'].apply(is_valid_species)
    else:
        prediction_df['is_mouse_or_rat_only'] = False

    # Apply classification only to valid rows
    def conditional_process(row):
        if row['is_mouse_or_rat_only']:
            return pd.Series(process_age_predictions(row['PMID'], row[pred_col_age]))
        else:
            return pd.Series([None, None])

    prediction_df[['age_classification', 'prediction_normalized_age']] = prediction_df.apply(
        conditional_process, axis=1
    )

    # Fill in unprocessed rows
    prediction_df['age_classification'] = prediction_df['age_classification'].fillna("not processed")
    prediction_df['prediction_normalized_age'] = prediction_df['prediction_normalized_age'].fillna(prediction_df[pred_col_age])

    prediction_df.drop(columns=["is_mouse_or_rat_only"], inplace=True)

    return prediction_df
    

def main():
    parser = argparse.ArgumentParser(
        description="Classify + map age predictions with species context and write a doc-level CSV."
    )
    parser.add_argument(
        "--model-name-str",
        default="age_unsloth_meta_llama_3.1_8b",
        help="Model name tag used in the output filename."
    )
    parser.add_argument(
        "--in-verified",
        default="./model_predictions/age/df_age_predictions_verified_20250722.csv",
        help="Input CSV with verified age predictions (must contain 'verified_prediction_encoded_label')."
    )
    parser.add_argument(
        "--species-csv",
        default="./model_predictions/regex/server/species_predictions.csv",
        help="CSV with species predictions used for mapping."
    )
    parser.add_argument(
        "--pred-col-species",
        default="prediction_encoded_label",
        help="Column in the main predictions file containing the original encoded label to compare/keep."
    )
    parser.add_argument(
        "--pred-col-age",
        default="verified_prediction_encoded_label",
        help="Column in the main predictions file containing the verified age label."
    )
    parser.add_argument(
        "--out-csv",
        default="./model_predictions/age/age_unsloth_meta_llama_3.1_8b_doc_level_predictions_mapped_20250722.csv",
        help="Output CSV path for the mapped, doc-level predictions."
    )

    args = parser.parse_args()

    # Load inputs
    in_path = Path(args.in_verified)
    species_path = Path(args.species_csv)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not species_path.is_file():
        raise FileNotFoundError(f"Species CSV not found: {species_path}")

    predictions_df = pd.read_csv(in_path)
    print(f"Loaded {len(predictions_df)} predictions from {in_path}")

    # Validate required columns
    if args.pred_col_age not in predictions_df.columns:
        raise ValueError(
            f"Input file '{in_path}' must contain '{args.pred_col_age}' column."
        )

    species_df = pd.read_csv(species_path)

    # Do the classification/mapping
    predictions_df = classify_age_prediction(
        predictions_df,
        pred_col_species=args.pred_col_species,
        pred_col_age=args.pred_col_age,
        species_df=species_df,
    )

    # Build/save output
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_path, index=False)

    print(
        f"Processed {len(predictions_df)} documents with age predictions.\n"
        f"Saved to {out_path}"
    )

if __name__ == "__main__":
    main()
