import pandas as pd
import os

def split_csv(input_files, output_dir, chunk_name_prefix="dict_mapped_", n_chunks=10):
    """
    Accepts one CSV file or a list of CSV files, combines them, 
    then splits the combined dataframe into n_chunks CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Handle list of input files ---
    if isinstance(input_files, str):
        input_files = [input_files]

    dfs = []
    for f in input_files:
        print(f"Loading {f} ...")
        dfs.append(pd.read_csv(f))

    # Combine
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total combined rows: {len(df)}")

    # --- Split into chunks ---
    total_rows = len(df)
    chunk_size = total_rows // n_chunks + (total_rows % n_chunks > 0)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk_df = df.iloc[start:end]

        chunk_path = os.path.join(output_dir, f"{chunk_name_prefix}ner_chunk_{i+1}.csv")
        chunk_df.to_csv(chunk_path, index=False)

        print(f"Saved chunk {i+1} with {len(chunk_df)} rows to {chunk_path}")
        
# Example usage
split_csv(
    input_files=[
        "03_IE_ner/data/animal_studies_with_drug_disease/animal_combined_filtered_df_non_empty_ner.csv"
        #"04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_2879.csv",
        #"04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_4489.csv",
        #"04_normalization/data/mapped_to_dict/aggregated_ner_annotations_basic_dict_mapped_595768.csv",
    ],
    output_dir="04_normalization/data/raw_ner/chunks/",
    chunk_name_prefix="",
    n_chunks=10
)