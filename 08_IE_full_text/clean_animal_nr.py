import pandas as pd
import re
from regex_classifiers.species_classifier import SpeciesClassifier

# Read document-level predictions (comma-separated values per document)
ner_df = pd.read_csv("08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions.csv")
text_df = pd.read_csv("07_full_text_retrieval/materials_methods/combined/combined_methods.csv")

# Merge predictions with full text
merged_df = ner_df.merge(text_df[['PMID', 'Text']], on='PMID', how='left')

species_clf = SpeciesClassifier()

def match_doc_level_predictions(row, context_window=50):
    """
    For each predicted number in a document, check its context for animal relevance.
    """
    prediction_str = row['prediction_encoded_label']
    text = row['Text'].lower() if 'Text' in row else ''
    if not isinstance(text, str) or pd.isna(prediction_str):
        return []

    values = [v.strip().lower() for v in str(prediction_str).split(',') if v.strip()]
    results = []

    for val in values:
        pattern = rf'\b{re.escape(val)}\b'
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]

            # Classify species context
            _, found_labels = species_clf.classify(context)
            is_animal_context = 1
            if len(found_labels) == 1 and found_labels[0] == "species-other":
                if not re.search(r'\banimals?\b', context, flags=re.IGNORECASE):
                    is_animal_context = 0
                else:
                    found_labels = ["animal_keyword"]

            results.append({
               
                "prediction_encoded_label_new": val,
                "start": start,
                "end": end,
                "context": context,
                "species_classification": found_labels,
                "is_animal_context": is_animal_context,
            })

    return results

# Apply
merged_df['extracted_spans'] = merged_df.apply(match_doc_level_predictions, axis=1)

# Flatten
exploded_df = merged_df.explode('extracted_spans').reset_index(drop=True)
extracted_cols = pd.json_normalize(exploded_df['extracted_spans']).reset_index(drop=True)
exploded_df = pd.concat([exploded_df.drop(columns=['extracted_spans']), extracted_cols], axis=1)


# Save
# Select only desired output columns (exclude Text/context/Source/etc.)
columns_to_keep = [
    "PMID",
    "prediction_encoded_label_new",
    "context",
    "species_classification",
    "is_animal_context"
]

final = exploded_df[columns_to_keep]
final = final.drop_duplicates(subset=["PMID", "prediction_encoded_label_new", "context"])

final.to_csv(
    "08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions_with_context.csv",
    index=False
)

# Filter to animal-relevant
final = final[final['is_animal_context'] == 1]

# Group and join the values into a single comma-separated string
grouped_df = (
    final.groupby("PMID")["prediction_encoded_label_new"]
    .apply(lambda x: ', '.join(sorted(set(x))))
    .reset_index()
    .rename(columns={"prediction_encoded_label_new": "prediction_encoded_label"})
)

grouped_df.to_csv(
    "08_IE_full_text/model_predictions/animals_nr/doc_animals_nr_predictions_clean.csv",
    index=False
)