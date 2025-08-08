import argparse
import pandas as pd
import os

def split_jsonl_into_chunks(input_path, num_chunks, output_dir):
    # Load the JSONL file
    print(f"Reading: {input_path}")
    df = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df)} rows.")

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Split into chunks
    chunks = [df.iloc[i::num_chunks] for i in range(num_chunks)]

    # Save each chunk
    for i, chunk in enumerate(chunks, 1):
        output_file = os.path.join(output_dir, f"chunk_{i}.jsonl")
        chunk.to_json(output_file, orient="records", lines=True)
        print(f"Saved chunk {i} with {len(chunk)} rows → {output_file}")

    print("Done.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into N evenly sized chunks."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=10,
        help="Number of chunks to split the file into (default: 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../07_full_text_retrieval/materials_methods/combined/split_chunks",
        help="Directory to save chunked files (default: split_chunks)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    split_jsonl_into_chunks(args.input_jsonl, args.num_chunks, args.output_dir)

if __name__ == "__main__":
    main()
