import pandas as pd
import ast
import re

def classify_age(age_in_weeks_str):
    """
    Classify age based on the input string.
    :param age_in_weeks_str: String representing age in weeks.
    :return: Age classification as 'young', 'adult', or 'aged'.
    """

    age_in_weeks = float(age_in_weeks_str)

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

def process_age_predictions(pred_str):
    """
    Process age predictions from a DataFrame column.
    
    :param pred_str: String representation of age predictions.
    :return: Age classification as 'young', 'adult', or 'aged'.
    """
    if not isinstance(pred_str, str):
        return "unknown"
    
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
    mapped_classificatinos = ", ".join(list(set(classifications)))
    mapped_normalized_age = ", ".join(list(set(normalized_age)))
    print(f"Mapped classifications: {mapped_classificatinos}")
    return mapped_classificatinos, mapped_normalized_age
    
    
def classify_age_prediction(prediction_df, pred_col="prediction_encoded_label", species_df=None):
    """
    Classify age predictions based on sentence-level species annotations.
    Only classify if supporting sentences contain only 'mouse', 'rat', or species-other.
    If any non-allowed species (other than species-other) is present, prediction is skipped.
    
    :param prediction_df: DataFrame with age predictions
    :param pred_col: Column with age prediction strings
    :param species_df: Sentence-level species annotations
    :return: DataFrame with age classifications
    """
    prediction_df = prediction_df.copy()

    if species_df is not None:
        species_df = species_df.copy()

        # Parse list-like strings to real lists
        species_df['prediction_encoded_label'] = species_df['prediction_encoded_label'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

        # Add sentence-level unique ID
        species_df['sentence_uid'] = species_df['PMID'].astype(str) + '_' + species_df['sentence_id'].astype(str)

        # Build map: sentence_uid → species label list
        sentence_species_map = dict(zip(species_df['sentence_uid'], species_df['prediction_encoded_label']))

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
            return pd.Series(process_age_predictions(row[pred_col]))
        else:
            return pd.Series([None, None])

    prediction_df[['age_classification', 'prediction_normalized_age']] = prediction_df.apply(
        conditional_process, axis=1
    )

    # Fill in unprocessed rows
    prediction_df['age_classification'] = prediction_df['age_classification'].fillna("not processed")
    prediction_df['prediction_normalized_age'] = prediction_df['prediction_normalized_age'].fillna(prediction_df[pred_col])

    prediction_df.drop(columns=["is_mouse_or_rat_only"], inplace=True)

    return prediction_df
    

def main():
    model_name_str = "age_unsloth_meta_llama_3.1_8b"
    input_file_name = f"08_IE_full_text/model_predictions/age/{model_name_str}_doc_level_predictions.csv"
    predictions_df = pd.read_csv(input_file_name)
    if "prediction_encoded_label" not in predictions_df.columns:    
        raise ValueError(f"Input file '{input_file_name}' must contain 'prediction_encoded_label' column.")
    
    species_df = pd.read_csv("08_IE_full_text/model_predictions/regex/species_predictions.csv")
    predictions_df = classify_age_prediction(predictions_df, pred_col="prediction_encoded_label", species_df=species_df)

    save_path= f"./08_IE_full_text/model_predictions/age/{model_name_str}_doc_level_predictions_mapped.csv"
    predictions_df.to_csv(save_path, index=False)
    print(f"Processed {len(predictions_df)} documents with age predictions. Saved to {save_path}")
    
if __name__ == "__main__":
    main()