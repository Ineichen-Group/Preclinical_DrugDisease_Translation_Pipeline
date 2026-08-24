import ast
import glob
import os
import re

import pandas as pd

from abbreviations import schwartz_hearst


NER_PREDICTION_COL = "ner_prediction_BioLinkBERT-base_normalized"


def remove_spaces_around_apostrophe_and_dash(text):
    """Normalize spacing around punctuation and repeated whitespace."""
    text = text.replace(" ' ", "'")
    text = text.replace("' s", "'s")
    text = text.replace(" - ", "-")
    text = text.replace("- ", "-")
    text = text.replace(" / ", "/")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace("[ ", "[")
    text = text.replace(" ]", "]")
    text = re.sub(r"\s+", " ", text)

    return text


def process_ner_predictions(directory_path):
    """Load and combine NER prediction CSV files."""
    dfs = []
    count_files = 0

    for filename in os.listdir(directory_path):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(directory_path, filename)

        df = pd.read_csv(file_path)[
            ["PMID", NER_PREDICTION_COL]
        ]

        dfs.append(df)
        count_files += 1

    if not dfs:
        raise RuntimeError(
            f"No CSV files found in NER prediction directory: {directory_path}"
        )

    df_pred_full = pd.concat(
        dfs,
        ignore_index=True,
    ).drop_duplicates()

    # Separate documents with and without extracted entities
    df_empty = df_pred_full[
        df_pred_full[NER_PREDICTION_COL].apply(
            lambda x: len(x) <= 2
        )
    ].copy()

    df_pred = df_pred_full[
        df_pred_full[NER_PREDICTION_COL].apply(
            lambda x: len(x) > 2
        )
    ].copy()

    df_pred[NER_PREDICTION_COL] = df_pred[
        NER_PREDICTION_COL
    ].apply(remove_spaces_around_apostrophe_and_dash)

    print(
        f"Read {count_files} files, "
        f"full df shape {df_pred_full.shape}, "
        f"df shape without empty NER {df_pred.shape}"
    )

    return df_pred, df_empty


def extract_abbreviation_definition_pairs(doc_text):
    """Extract abbreviation-definition pairs using Schwartz-Hearst."""
    return schwartz_hearst.extract_abbreviation_definition_pairs(
        doc_text=doc_text
    )


def extract_abbreviation_from_full_text(
    pmid_set,
    folder_path=(
        "../02_animal_study_classification/"
        "data/animal_studies_for_ner/update_2025"
    ),
    save_to_path=(
        "./data/abbreviations_expansion/"
        "pmid_abbreviations.csv"
    ),
):
    """Extract abbreviation definitions for the requested PMIDs."""
    csv_files = glob.glob(
        os.path.join(folder_path, "*.csv")
    )

    print(
        f"Found {len(csv_files)} full-text CSV files "
        f"in {folder_path}"
    )

    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in full-text directory: {folder_path}"
        )

    # Normalize PMID type for matching
    pmid_set = {
        str(pmid).strip()
        for pmid in pmid_set
    }

    # Ensure destination directory exists
    save_parent_dir = os.path.dirname(save_to_path)

    if save_parent_dir:
        os.makedirs(
            save_parent_dir,
            exist_ok=True,
        )

    # Remove an incomplete file from an earlier extraction attempt
    if os.path.isfile(save_to_path):
        os.remove(save_to_path)

    count_files = 0
    count_matching_articles = 0

    for file in csv_files:
        if count_files in [100, 200, 300, 400, 500, 600]:
            print(
                f"Processing reached {count_files} "
                f"with {file}"
            )

        df = pd.read_csv(file)
        count_files += 1

        if "PMID" not in df.columns:
            continue

        if "Text" not in df.columns:
            continue

        # Normalize PMID type in full-text data
        df["PMID"] = (
            df["PMID"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        filtered_df = df[
            df["PMID"].isin(pmid_set)
        ].copy()

        if filtered_df.empty:
            continue

        count_matching_articles += len(filtered_df)

        filtered_df[
            "abbreviation_definition_pairs"
        ] = filtered_df["Text"].apply(
            extract_abbreviation_definition_pairs
        )

        columns_to_save = [
            "PMID",
            "abbreviation_definition_pairs",
        ]

        filtered_df[
            columns_to_save
        ].to_csv(
            save_to_path,
            mode="a",
            header=not os.path.exists(save_to_path),
            index=False,
        )

    print(
        f"Completed reading {count_files} full-text files."
    )
    print(
        f"Found {count_matching_articles} matching articles."
    )

    if not os.path.isfile(save_to_path):
        raise RuntimeError(
            "No abbreviation file was created. "
            "No matching PMIDs were found.\n"
            f"Full-text directory: {folder_path}\n"
            f"Number of requested PMIDs: {len(pmid_set)}"
        )

    print(
        f"Abbreviations saved to {save_to_path}"
    )

    return save_to_path


def parse_abbreviation_dict(value):
    """Convert stored abbreviation dictionary strings back to dictionaries."""
    if pd.isna(value):
        return {}

    if isinstance(value, dict):
        return value

    try:
        parsed = ast.literal_eval(value)
    except (
        ValueError,
        SyntaxError,
        TypeError,
    ):
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


def load_abbreviations_from_csv(
    save_abbrev_to_path,
    pmid_set,
    full_text_dir=(
        "../02_animal_study_classification/"
        "data/animal_studies_for_ner/update_2025"
    ),
):
    """Load abbreviations or extract them if the file does not exist."""
    if not os.path.isfile(save_abbrev_to_path):
        print(
            "Abbreviations file not found at "
            f"{save_abbrev_to_path}"
        )

        print(
            f"Extracting abbreviations for "
            f"{len(pmid_set)} PMIDs..."
        )

        extract_abbreviation_from_full_text(
            pmid_set=pmid_set,
            folder_path=full_text_dir,
            save_to_path=save_abbrev_to_path,
        )

    else:
        print(
            "Loading existing abbreviations from "
            f"{save_abbrev_to_path}"
        )

    if not os.path.isfile(save_abbrev_to_path):
        raise FileNotFoundError(
            "Abbreviation file was not created: "
            f"{save_abbrev_to_path}"
        )

    abbrev_df = pd.read_csv(
        save_abbrev_to_path
    )

    required_columns = {
        "PMID",
        "abbreviation_definition_pairs",
    }

    missing_columns = (
        required_columns
        - set(abbrev_df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Abbreviation file is missing "
            f"columns: {missing_columns}"
        )

    abbrev_df["PMID"] = (
        abbrev_df["PMID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    abbrev_df[
        "abbreviation_definition_pairs"
    ] = abbrev_df[
        "abbreviation_definition_pairs"
    ].apply(parse_abbreviation_dict)

    print(
        f"Loaded abbreviations for "
        f"{len(abbrev_df)} PMIDs"
    )

    return abbrev_df


def extract_unique_entities(
    nct_id,
    annotation_list,
    abbreviation_definition_pairs,
    model="linkbert",
):
    """Extract unique disease and drug entities from NER predictions."""
    unique_conditions = set()
    unique_interventions = set()
    interventions_type = set()

    try:
        annotation_list = ast.literal_eval(
            annotation_list
        )
    except (
        ValueError,
        SyntaxError,
        TypeError,
    ) as e:
        print(
            f"Issue parsing NER predictions "
            f"for PMID {nct_id}"
        )
        print(annotation_list)
        print("Error:", e)

        return "", "", ""

    if not isinstance(
        abbreviation_definition_pairs,
        dict,
    ):
        abbreviation_definition_pairs = {}

    for annotation in annotation_list:
        _, _, entity_type, entity_name = annotation

        # Skip potentially malformed tokenized entities
        if entity_name.startswith("##"):
            continue

        # Skip empty or single-character entities
        if not entity_name or len(entity_name) == 1:
            continue

        # Replace abbreviations with full forms
        if entity_name in abbreviation_definition_pairs:
            entity_name = abbreviation_definition_pairs[
                entity_name
            ]

        elif entity_name.upper() in abbreviation_definition_pairs:
            entity_name = abbreviation_definition_pairs[
                entity_name.upper()
            ]

        entity_name = entity_name.lower()

        if entity_type == "DISEASE":
            unique_conditions.add(entity_name)

        elif entity_type == "DRUG":
            unique_interventions.add(entity_name)
            interventions_type.add(entity_type)

    return (
        "|".join(list(unique_conditions)),
        "|".join(list(unique_interventions)),
        "|".join(list(interventions_type)),
    )


def get_empty_ner_stats(df_pred):
    """Calculate statistics for articles with missing extracted entities."""
    condition_col = "unique_conditions_linkbert_predictions"
    intervention_col = "unique_interventions_linkbert_predictions"

    empty_interventions = (
        df_pred[intervention_col]
        .apply(
            lambda x: (
                isinstance(x, str)
                and x.strip() == ""
            )
        )
        .sum()
    )

    empty_conditions = (
        df_pred[condition_col]
        .apply(
            lambda x: (
                isinstance(x, str)
                and not x
            )
        )
        .sum()
    )

    both_empty = df_pred[
        df_pred[condition_col].apply(
            lambda x: (
                isinstance(x, str)
                and not x
            )
        )
        & df_pred[intervention_col].apply(
            lambda x: (
                isinstance(x, str)
                and x.strip() == ""
            )
        )
    ].shape[0]

    results_empty_entities = pd.DataFrame(
        {
            "Empty unique_conditions_linkbert_predictions": [
                empty_conditions
            ],
            "Empty unique_interventions_linkbert_predictions": [
                empty_interventions
            ],
            "Both Empty": [
                both_empty
            ],
        }
    )

    print(results_empty_entities)

    stats_dir = "./ner_stats"

    os.makedirs(
        stats_dir,
        exist_ok=True,
    )

    results_empty_entities.to_csv(
        os.path.join(
            stats_dir,
            f"empty_ner_predictions_count_"
            f"{len(results_empty_entities)}.csv",
        ),
        index=False,
    )


def main():
    # NER predictions for the 2025 update
    folder_with_ner_prediction = (
        "./model_predictions/update_2025/drug_disease"
    )

    df_pred, df_empty = process_ner_predictions(
        folder_with_ner_prediction
    )

    # Normalize PMID type
    df_pred["PMID"] = (
        df_pred["PMID"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    pmid_set = set(df_pred["PMID"])

    print(
        f"Number of PMIDs with NER predictions: "
        f"{len(pmid_set)}"
    )

    # Abbreviation extraction/loading
    save_abbrev_to_path = (
        "./data/abbreviations_expansion/"
        f"pmid_abbreviations_{len(pmid_set)}_update_2025.csv"
    )

    full_text_dir = (
        "../02_animal_study_classification/"
        "data/animal_studies_for_ner/update_2025"
    )

    abbrev_df = load_abbreviations_from_csv(
        save_abbrev_to_path=save_abbrev_to_path,
        pmid_set=pmid_set,
        full_text_dir=full_text_dir,
    )

    # Join abbreviations to NER predictions
    df_pred_with_abbrev = df_pred.merge(
        abbrev_df,
        on="PMID",
        how="left",
    )

    # PMIDs without an abbreviation entry should have an empty dictionary
    df_pred_with_abbrev[
        "abbreviation_definition_pairs"
    ] = df_pred_with_abbrev[
        "abbreviation_definition_pairs"
    ].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    print(
        f"Abbreviations: {abbrev_df.shape}, "
        f"joined: {df_pred_with_abbrev.shape}"
    )

    # Extract unique conditions and interventions
    print("Extracting unique entities...")

    model_name_str_biolink = "linkbert"

    condition_col = (
        f"unique_conditions_"
        f"{model_name_str_biolink}_predictions"
    )
    intervention_col = (
        f"unique_interventions_"
        f"{model_name_str_biolink}_predictions"
    )

    (
        df_pred_with_abbrev[condition_col],
        df_pred_with_abbrev[intervention_col],
        _,
    ) = zip(
        *df_pred_with_abbrev.apply(
            lambda row: extract_unique_entities(
                row["PMID"],
                row[NER_PREDICTION_COL],
                row["abbreviation_definition_pairs"],
            ),
            axis=1,
        )
    )

    get_empty_ner_stats(
        df_pred_with_abbrev
    )

    # Keep articles containing both a condition and an intervention
    filtered_df_non_empty = df_pred_with_abbrev[
        df_pred_with_abbrev[
            condition_col
        ].apply(
            lambda x: (
                isinstance(x, str)
                and bool(x)
            )
        )
        & df_pred_with_abbrev[
            intervention_col
        ].apply(
            lambda x: (
                isinstance(x, str)
                and x.strip() != ""
            )
        )
    ]

    print(
        "Articles with both conditions and interventions: "
        f"{filtered_df_non_empty.shape}"
    )

    # Save final results
    save_dir = (
        "./data/animal_studies_with_drug_disease"
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    df_to_save = filtered_df_non_empty[
        [
            "PMID",
            condition_col,
            intervention_col,
        ]
    ].drop_duplicates()

    print(
        "df_to_save shape after dropping duplicates: "
        f"{df_to_save.shape}"
    )

    save_file_name = (
        f"filtered_df_non_empty_"
        f"{len(df_to_save)}_update_2025"
    )

    df_to_save.to_csv(
        os.path.join(
            save_dir,
            f"{save_file_name}.csv",
        ),
        index=False,
    )

    df_to_save[["PMID"]].to_csv(
        os.path.join(
            save_dir,
            f"{save_file_name}_PMIDs.csv",
        ),
        index=False,
    )

    print(
        f"Saved final results to {save_dir}"
    )


if __name__ == "__main__":
    main()