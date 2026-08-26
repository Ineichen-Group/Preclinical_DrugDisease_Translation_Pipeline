import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge normalized preclinical drug and disease entity data "
            "into one PMID-level CSV."
        )
    )

    parser.add_argument(
        "--drug_input",
        required=True,
        help="Path to preclinical drug CSV.",
    )

    parser.add_argument(
        "--disease_input",
        required=True,
        help="Path to preclinical disease CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to merged output CSV.",
    )

    parser.add_argument(
        "--doc_id_col",
        default="PMID",
    )

    # Drug columns
    parser.add_argument(
        "--drug_raw_col",
        default="unique_interventions_linkbert_predictions",
    )
    parser.add_argument(
        "--drug_linked_col",
        default="drug_umls_term_norm",
    )
    parser.add_argument(
        "--drug_linked_id_col",
        default="drug_umls_termid",
    )
    parser.add_argument(
        "--drug_parent_col",
        default="nearest_dataset_parent_umls_label",
    )
    parser.add_argument(
        "--drug_merged_col",
        default="merged_umls_label",
    )
    parser.add_argument(
        "--drug_merged_id_col",
        default="merged_umls_termid",
    )

    # Disease columns
    parser.add_argument(
        "--disease_raw_col",
        default="unique_conditions_linkbert_predictions",
    )
    parser.add_argument(
        "--disease_linked_col",
        default="disease_mondo_term_norm",
    )
    parser.add_argument(
        "--disease_linked_id_col",
        default="disease_mondo_termid",
    )
    parser.add_argument(
        "--disease_clean_col",
        default="disease_term_mondo_clean",
    )
    parser.add_argument(
        "--disease_parent_col",
        default="nearest_dataset_parent_label",
    )
    parser.add_argument(
        "--disease_merged_col",
        default="merged_mondo_label",
    )
    parser.add_argument(
        "--disease_merged_id_col",
        default="merged_mondo_termid",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading drug data: {args.drug_input}")
    df_drugs = pd.read_csv(args.drug_input)

    print(f"Drug shape: {df_drugs.shape}")

    drug_cols = [
        args.doc_id_col,
        args.drug_raw_col,
        args.drug_linked_col,
        args.drug_linked_id_col,
        args.drug_parent_col,
        args.drug_merged_col,
        args.drug_merged_id_col,
    ]

    missing_drug_cols = [
        col for col in drug_cols
        if col not in df_drugs.columns
    ]

    if missing_drug_cols:
        raise ValueError(
            f"Missing drug columns: {missing_drug_cols}"
        )

    df_drugs = df_drugs[
        drug_cols
    ].copy()

    print(f"Reading disease data: {args.disease_input}")
    df_diseases = pd.read_csv(args.disease_input)

    print(f"Disease shape: {df_diseases.shape}")

    disease_cols = [
        args.doc_id_col,
        args.disease_raw_col,
        args.disease_linked_col,
        args.disease_linked_id_col,
        args.disease_clean_col,
        args.disease_parent_col,
        args.disease_merged_col,
        args.disease_merged_id_col,
    ]

    missing_disease_cols = [
        col for col in disease_cols
        if col not in df_diseases.columns
    ]

    if missing_disease_cols:
        raise ValueError(
            f"Missing disease columns: {missing_disease_cols}"
        )

    df_diseases = df_diseases[
        disease_cols
    ].copy()

    print("Merging drug and disease data...")

    df_merged = df_diseases.merge(
        df_drugs,
        on=args.doc_id_col,
        how="inner",
    )

    print(f"Merged shape: {df_merged.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_merged.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved merged CSV to: {output_path}")


if __name__ == "__main__":
    main()