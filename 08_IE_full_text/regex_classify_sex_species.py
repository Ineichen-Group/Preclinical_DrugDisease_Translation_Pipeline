import pandas as pd
import os
import re
import ast
import pandas as pd
import re
import time

SPECIES_LABELS = ["mouse", "rat", "rabbit", "monkey", "guinea pig", "species-other", "dog", "cat", "pig"]
SEX_LABELS = {
    0: "sex-both",
    1: "sex-female",
    2: "sex-male",
    3: "sex-not-reported"
}

REGEX_DICT = {
    "sex": {
        "sex-female": [r"\bfemales?\b", r"\bmaternal\b", r"\bmothers?\b", r"\bsisters?\b"],
        "sex-male": [r"\bmales?\b", r"\bfathers?\b", r"\bbrothers?\b"],
        "sex-both": [r"\beither sex(?:es)\b", r"\both sex(?:es)\b", r"\b(?:of|the)\seither sex\b"]
        },
    "species": {
        "rat": [r"\brats?\b"],
        "mouse": [r"\bmouse\b", r"\bmice\b"],
        "cat": [r"\bcats?\b"],
        "dog": [r"\bdogs?\b"],
        "guinea pig": [r"\bguinea pig\s?\b", r"\bguinea\b"],
        "monkey": [r"\bmonkeys?\b", r"\bmacaques?\b", r"\bchimpanzees?\b", r"\borangutans?\b", r"\bbonoboss?\b", r"\bgibbons?\b"],
        "pig": [r"\bpigs?\b", r"\bswines?\b", r"\bpiglets?\b"],
        "rabbit": [r"\brabbits?\b"],
        "species-other": []
        }
}

def label_to_vector(label, species_labels):
    # Initialize the binary vector with zeros
    label_vector = [0] * len(species_labels)
    
    for species in label:
        if species in species_labels:
            idx = species_labels.index(species)
            label_vector[idx] = 1
            
    return label_vector

def classify_sex(text):
    matches = {label: False for label in REGEX_DICT["sex"]}

    for label, patterns in REGEX_DICT["sex"].items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches[label] = True

    if matches["sex-female"] and matches["sex-male"]:
        label = "sex-both"
    elif matches["sex-female"]:
        label = "sex-female"
    elif matches["sex-male"]:
        label = "sex-male"
    else:
        label = "sex-not-reported"
        
    # Now find the number
    num = [k for k, v in SEX_LABELS.items() if v == label][0]

    return num, label

FALSE_CONTEXT_TERMS = r"""
    \b(
        antibody|antibodies|antiserum|
        monoclonal|polyclonal|
        Ig\s*[A-Z]{1,2}|
        mAb|pAb|HRP|APC|FITC|PE|Cy\d+|
        ELISA|immunoblot|western blot|immunostaining|
        conjugated|biotinylated|fluorescent-labeled|
        GAPDH|tubulin|β-actin|
        luciferase|peroxidase|polymerase|qPCR|RT-PCR|Taq|
        serum|lysate|recombinant|TG2|anti|OX\d+|CD\d+
    )\b
"""
WINDOW = 10

def is_in_false_context(text, match_start, match_end, context_terms, window=5):
    # Tokenize with positions
    tokens = list(re.finditer(r'\b\w+\b', text))
    
    # Find token index of species match
    match_token_indices = [
        i for i, tok in enumerate(tokens)
        if match_start <= tok.start() < match_end
    ]

    if not match_token_indices:
        return False  # Fallback if match not mapped properly

    # For each species token index, check ±window
    for idx in match_token_indices:
        start = max(0, idx - window)
        end = min(len(tokens), idx + window + 1)
        context_window = tokens[start:end]
        for tok in context_window:
            if re.search(context_terms, tok.group(), re.IGNORECASE | re.VERBOSE):
                return True  # Found context term nearby

    return False

def classify_species(text):
    matched_labels = []
    species_vector = [0] * len(SPECIES_LABELS)

    for idx, label in enumerate(SPECIES_LABELS):
        patterns = REGEX_DICT["species"].get(label, [])
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if not is_in_false_context(text, match.start(), match.end(), FALSE_CONTEXT_TERMS):
                    species_vector[idx] = 1
                    matched_labels.append(label)
                    break  # one match is enough per label

    if not matched_labels:
        species_vector[SPECIES_LABELS.index("species-other")] = 1
        matched_labels = ["species-other"]

    return species_vector, list(set(matched_labels))


def main(df_path, category, text_col, output_dir):
    df = pd.read_csv(df_path)
    df[text_col] = df[text_col].str.replace(r'\s+', ' ', regex=True).str.strip()

    if category == "sex":
        df[['prediction_encoded_num', 'prediction_encoded_label']] = df[text_col].apply(lambda x: pd.Series(classify_sex(x)))
        
    elif category == "species":
        df[['prediction_encoded_num', 'prediction_encoded_label']] = df[text_col].apply(lambda x: pd.Series(classify_species(x)))

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    df[['PMID','prediction_encoded_num', 'prediction_encoded_label']].to_csv(f"{output_dir}/{category}_predictions_MS.csv", index=False)

    print(f"RegEx classification ran successfully. The output is saved to {output_dir}")


if __name__=="__main__":
    
    df_path = "07_full_text_retrieval/materials_methods/combined/combined_methods.csv"
    category = "species"  # "sex" or "species"
    output_dir = f"08_IE_full_text/model_predictions/{category}"
    text_col = "Text"
    
    start_time = time.time()
    main(df_path, category, text_col, output_dir)
    end_time = time.time()

    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)

    print(f"\nDone in {elapsed:.2f} seconds ({int(mins)}m {secs:.2f}s).")