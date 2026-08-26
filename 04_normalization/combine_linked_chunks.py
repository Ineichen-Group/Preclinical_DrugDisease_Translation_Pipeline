from pathlib import Path
import argparse
import re
import pandas as pd


OUTPUT_DIR = Path(
    "/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/"
    "mapped_to_embeddings_ontologies"
)


def combine_chunks(chunks_dir, name, pattern, output_file):
    chunk_files = []

    for f in chunks_dir.iterdir():
        match = pattern.fullmatch(f.name)

        if match:
            chunk_id = int(match.group(1))
            chunk_files.append((chunk_id, f))

    if not chunk_files:
        raise FileNotFoundError(
            f"No matching {name} chunk files found in {chunks_dir}"
        )

    chunk_files.sort(key=lambda x: x[0])

    print(f"\n{name.upper()}")
    print(f"Found {len(chunk_files)} chunk files")

    dfs = []

    for chunk_id, f in chunk_files:
        print(f"Reading chunk {chunk_id}: {f.name}")
        dfs.append(pd.read_csv(f))

    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined shape: {combined_df.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    print(f"Saved combined CSV to: {output_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks_dir",
        required=True,
        help="Directory containing drug and disease normalization chunks.",
    )

    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for output files, e.g. update_2025.",
    )

    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)

    suffix = f"_{args.suffix}" if args.suffix else ""

    datasets = {
        "drug": {
            "pattern": re.compile(
                r"drug_mapped_preclinical_drug(?:_enriched)?_(\d+)\.csv"
            ),
            "output": OUTPUT_DIR
            / f"drug_mapped_preclinical_enriched_all{suffix}.csv",
        },
        "disease": {
            "pattern": re.compile(
                r"disease_mapped_preclinical_disease(?:_enriched)?_(\d+)\.csv"
            ),
            "output": OUTPUT_DIR
            / f"disease_mapped_preclinical_enriched_all{suffix}.csv",
        },
    }

    for name, config in datasets.items():
        combine_chunks(
            chunks_dir=chunks_dir,
            name=name,
            pattern=config["pattern"],
            output_file=config["output"],
        )


if __name__ == "__main__":
    main()