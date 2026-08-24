import argparse
import os

import pandas as pd


def split_csv(
    input_files,
    output_dir,
    chunk_name_prefix="",
    n_chunks=10,
):
    """
    Accept one CSV file or multiple CSV files, combine them,
    and split the combined dataframe into n_chunks CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(input_files, str):
        input_files = [input_files]

    dfs = []

    for file in input_files:
        print(f"Loading {file} ...")
        dfs.append(pd.read_csv(file))

    df = pd.concat(
        dfs,
        ignore_index=True,
    )

    print(f"Total combined rows: {len(df)}")

    total_rows = len(df)
    chunk_size = (
        total_rows // n_chunks
        + (total_rows % n_chunks > 0)
    )

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(
            start + chunk_size,
            total_rows,
        )

        chunk_df = df.iloc[start:end]

        chunk_path = os.path.join(
            output_dir,
            f"{chunk_name_prefix}ner_chunk_{i + 1}.csv",
        )

        chunk_df.to_csv(
            chunk_path,
            index=False,
        )

        print(
            f"Saved chunk {i + 1} "
            f"with {len(chunk_df)} rows "
            f"to {chunk_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Split NER CSV files into chunks."
    )

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more input CSV files.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where chunks will be saved.",
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for chunk filenames.",
    )

    parser.add_argument(
        "--n_chunks",
        type=int,
        default=10,
        help="Number of chunks to create.",
    )

    args = parser.parse_args()

    split_csv(
        input_files=args.input_files,
        output_dir=args.output_dir,
        chunk_name_prefix=args.prefix,
        n_chunks=args.n_chunks,
    )


if __name__ == "__main__":
    main()