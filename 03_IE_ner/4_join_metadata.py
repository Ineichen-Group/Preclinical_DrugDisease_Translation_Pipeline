import argparse

import pandas as pd


PATH_TO_ANNOTATED_NER = (
    "./data/animal_studies_with_drug_disease"
)


def save_filtered_metadata():
    """Create metadata file for the original dataset."""

    metadata_file = (
        "02_animal_study_classification/data/animal_studies/"
        "full_pubmed_filtered_animal_6002827_metadata.csv"
    )

    metadata_full = pd.read_csv(metadata_file)
    metadata_full = metadata_full.drop_duplicates()
    metadata_full["PMID"] = (
        metadata_full["PMID"]
        .astype(str)
        .str.strip()
    )

    included_studies_main = pd.read_csv(
        f"{PATH_TO_ANNOTATED_NER}/filtered_df_non_empty_595768.csv"
    )
    included_studies_extra = pd.read_csv(
        f"{PATH_TO_ANNOTATED_NER}/filtered_df_non_empty_2879.csv"
    )

    included_studies = pd.concat(
        [
            included_studies_main,
            included_studies_extra,
        ],
        ignore_index=True,
    )

    included_studies["PMID"] = (
        included_studies["PMID"]
        .astype(str)
        .str.strip()
    )

    result = pd.merge(
        included_studies,
        metadata_full,
        how="left",
        on="PMID",
    )

    output_file = (
        f"{PATH_TO_ANNOTATED_NER}/"
        f"animal_studies_metadata_{len(result)}.csv"
    )

    print(f"Studies metadata: {result.shape}")

    result.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved filtered metadata with {len(result)} entries "
        f"to {output_file}"
    )


def save_filtered_metadata_update_2025():
    """Create metadata file for the 2025 update."""

    metadata_file = (
        "/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/02_animal_study_classification/data/animal_studies/"
        "update_2025/"
        "full_pubmed_filtered_animal_223396_metadata.csv"
    )

    metadata_full = pd.read_csv(metadata_file)
    metadata_full = metadata_full.drop_duplicates()
    metadata_full["PMID"] = (
        metadata_full["PMID"]
        .astype(str)
        .str.strip()
    )

    included_studies = pd.read_csv(
        f"{PATH_TO_ANNOTATED_NER}/"
        "filtered_df_non_empty_23796_update_2025.csv"
    )

    included_studies["PMID"] = (
        included_studies["PMID"]
        .astype(str)
        .str.strip()
    )

    result = pd.merge(
        included_studies,
        metadata_full,
        how="left",
        on="PMID",
    )

    output_file = (
        f"{PATH_TO_ANNOTATED_NER}/"
        f"animal_studies_metadata_{len(result)}_update_2025.csv"
    )

    print(f"Studies metadata: {result.shape}")

    result.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved filtered metadata with {len(result)} entries "
        f"to {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Save metadata for included animal studies."
    )

    parser.add_argument(
        "--dataset",
        choices=["original", "update_2025"],
        default="original",
        help="Dataset to process.",
    )

    args = parser.parse_args()

    if args.dataset == "original":
        save_filtered_metadata()

    elif args.dataset == "update_2025":
        save_filtered_metadata_update_2025()


if __name__ == "__main__":
    main()