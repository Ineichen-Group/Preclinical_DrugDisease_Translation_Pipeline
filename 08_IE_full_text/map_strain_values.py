import pandas as pd
import argparse
import os
from rapidfuzz import process, fuzz
from tqdm import tqdm
import glob
tqdm.pandas()

mapping_log = []  # global log list

def build_lookup_table(synonyms_file: str):
    """
    Load synonyms and create a canonical lookup dict (case-insensitive).
    """
    df = pd.read_csv(synonyms_file)
    syn_to_can_raw = df.set_index("Synonym")["StrainName"].to_dict()
    canonical_lookup = {str(k).lower(): v for k, v in syn_to_can_raw.items()}
    canonical_lookup.update({str(v).lower(): v for v in syn_to_can_raw.values()})
    return canonical_lookup

def map_strain(s, canonical_lookup, choices, cutoff=93):
    if pd.isna(s):
        mapping_log.append((s, s))
        return s

    s_str = str(s)
    if s_str.endswith("-") or s_str.endswith("/"):
        s_str = s_str[:-1]
    s_lower = s_str.lower()

    if s_lower in canonical_lookup:
        result = canonical_lookup[s_lower]
    else:
        best, score, _ = process.extractOne(
            s_lower, choices, scorer=fuzz.WRatio, score_cutoff=cutoff
        ) or (None, None, None)
        result = canonical_lookup.get(best, best) if best else s_str

    if result.lower().endswith("j") and not s_lower.endswith("j"):
        result = result[:-1]

    mapping_log.append((s_str, result))
    return result

def normalize_animal_strains(df, canonical_lookup, column="animal_strain", new_column="animal_strain_norm", delimiter="|"):
    choices = list(canonical_lookup.keys())

    def normalize_entry(entry):
        if pd.isna(entry):
            return entry
        strains = [s.strip() for s in str(entry).split(delimiter)]
        normalized = [map_strain(s, canonical_lookup, choices) for s in strains]
        unique_normalized = list(dict.fromkeys(normalized))
        return delimiter.join(unique_normalized)

    df[new_column] = df[column].progress_apply(normalize_entry)
    return df

def main():
    parser = argparse.ArgumentParser(description="Normalize animal strain names using fuzzy matching.")
    parser.add_argument("--input_csv", help="Path to input CSV with strain column.")
    parser.add_argument("--input_dir", help="Directory containing chunk_*.csv files to merge and process.")
    parser.add_argument("--output_csv", required=True, help="Path to save normalized output.")
    parser.add_argument("--synonyms_file", default="./data/strain_normalization/combined_df_mice_rat.csv", help="Path to synonyms CSV.")
    parser.add_argument("--column", default="prediction_encoded_label", help="Column name in input CSV to normalize.")
    parser.add_argument("--new_column", default="animal_strain_norm", help="Name of column for normalized output.")
    parser.add_argument("--delimiter", default=",", help="Delimiter for multiple strains in one cell.")
    args = parser.parse_args()

    # Load input data
    if args.input_dir:
        print(f"Reading all chunk_*.csv files from: {args.input_dir}")
        files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*.csv")))
        if not files:
            raise FileNotFoundError(f"No chunk_*.csv files found in {args.input_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Loaded {len(files)} files with total {len(df)} rows.")
    elif args.input_csv:
        print(f"Loading input CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
    else:
        raise ValueError("Either --input_csv or --input_dir must be specified.")

    # Load synonym table
    print(f"Loading synonym lookup from: {args.synonyms_file}")
    canonical_lookup = build_lookup_table(args.synonyms_file)

    # Normalize
    print(f"Normalizing strain names in column: {args.column}")
    df = normalize_animal_strains(df, canonical_lookup, column=args.column, new_column=args.new_column, delimiter=args.delimiter)

    # Save
    print(f"Saving to: {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
