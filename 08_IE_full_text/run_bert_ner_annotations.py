import os
import sys
import datetime
import argparse
import time
import pandas as pd
# Ensure core module is accessible
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(parent_dir)

from core.models import NERModel


def validate_path(path, is_directory=False):
    """
    Validates the existence of a file or directory.

    Args:
        path (str): Path to validate.
        is_directory (bool): True if the path should be a directory, False for files.
    
    Returns:
        str: Validated path.
    
    Raises:
        ValueError: If the path is invalid.
    """
    if is_directory and not os.path.isdir(path):
        raise ValueError(f"Directory does not exist: {path}")
    elif not is_directory and not os.path.isfile(path):
        raise ValueError(f"File does not exist: {path}")
    return path



def _load_test_data(path, sep=","):
    path = str(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def _split_into_n_chunks(df, n=10):
    n = max(1, int(n))
    if len(df) == 0:
        return []
    chunk_size = (len(df) + n - 1) // n  # ceil
    return [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

def run_inference(
    model_name,
    model_path,
    test_data_path,
    output_dir,
    output_file_suffix,
    text_column="Text",
    train_data_path=None,
    use_bio_format=True,
    custom_entity_grouping=False,
):
    """
    Run inference using a HuggingFace model and save annotations.

    Args:
        model_name (str): Name of the HuggingFace model.
        model_path (str): Path to the model directory.
        train_data_path (str): Path to the training data JSON.
        test_data_path (str): Path to the test data CSV/JSON/JSONL (tuple mode).
        output_dir (str): Directory to save output annotations.
        output_file_suffix (str): Suffix to append to output file name.
        text_column (str): Column containing text to annotate (tuple mode).
        use_bio_format (bool): Whether to save output in BIO format.
        custom_entity_grouping (bool): Whether to use custom entity grouping.
    """


    os.makedirs(output_dir, exist_ok=True)

    print("Starting inference...")
    print(f"Model name: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Input path: {test_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output file suffix: {output_file_suffix}")
    print(f"BIO format: {use_bio_format}")
    print(f"Custom entity grouping: {custom_entity_grouping}")

    # Initialize model
    model = NERModel(
        "huggingface",
        model_name,
        model_path,
        use_custom_entities_grouping=custom_entity_grouping,
    )

    print("Model initialized successfully.")

    model_name_clean = model_name.split("/")[-1]

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_file_format = (
        f"annotated_{model_name_clean}_{'BIO' if use_bio_format else 'tuples'}_{output_file_suffix}.csv"
    )
    output_path = os.path.join(output_dir, output_file_format)

    start_time = time.time()

    # Run inference
    if use_bio_format:
        print(f"Running inference in BIO format for {model_name}...")
        predictions = model.bert_predict_bio_format(
            train_data_path, test_data_path, "tokens", "ner_tags"
        )

        # Keep your original behavior
        if text_column in predictions.columns:
            predictions = predictions.drop(columns=[text_column])

        predictions.to_csv(output_path, index=False, sep=",")
        print(f"Annotations saved to {output_path}")

    else:
        print(f"Running inference in tuple format for {model_name}...")

        # Load test data from CSV/JSON/JSONL
        test_df = _load_test_data(test_data_path, sep=",")

        if text_column not in test_df.columns:
            raise KeyError(
                f"'{text_column}' column not found in test data. Columns: {list(test_df.columns)}"
            )

        # Split into 10 chunks
        chunks = _split_into_n_chunks(test_df, n=10)
        print(f"Loaded {len(test_df)} rows. Processing {len(chunks)} chunks sequentially...")

        # Directory for intermediate chunk outputs
        tmp_dir = os.path.join(output_dir, "tmp_chunks")
        os.makedirs(tmp_dir, exist_ok=True)

        chunk_paths = []
        for i, chunk_df in enumerate(chunks, start=1):
            if len(chunk_df) == 0:
                continue

            chunk_out = os.path.join(
                tmp_dir,
                f"chunk_{i:02d}_of_{len(chunks):02d}_{model_name_clean}_tuples_{output_file_suffix}.csv",
            )

            # Resume capability (skip if already computed)
            if os.path.exists(chunk_out):
                print(f"[Chunk {i}/{len(chunks)}] Found existing output, skipping: {chunk_out}")
                chunk_paths.append(chunk_out)
                continue

            print(f"[Chunk {i}/{len(chunks)}] Running inference on {len(chunk_df)} rows... \n")

            # annotate() accepts DF directly in the file_path parameter
            chunk_pred = model.annotate(chunk_df, source_column=text_column)

            # Drop original text column to match original output behavior
            if text_column in chunk_pred.columns:
                chunk_pred = chunk_pred.drop(columns=[text_column])

            # Save intermediate result
            chunk_pred.to_csv(chunk_out, index=False, sep=",")
            print(f"[Chunk {i}/{len(chunks)}] Saved: {chunk_out} \n")

            chunk_paths.append(chunk_out)

        if not chunk_paths:
            raise RuntimeError("No chunks were processed; check input data.")

        print(f"Combining {len(chunk_paths)} chunk files...")
        predictions = pd.concat(
            (pd.read_csv(p, sep=",") for p in chunk_paths),
            ignore_index=True,
        )

        # Final save
        predictions.to_csv(output_path, index=False, sep=",")
        print(f"Annotations saved to {output_path}")

    end_time = time.time()

    # Time breakdown
    elapsed_seconds = end_time - start_time
    elapsed_hours = elapsed_seconds / 3600

    print(
        f"Total runtime: {elapsed_hours:.2f} hours "
        f"({elapsed_seconds/60:.1f} minutes, {elapsed_seconds:.0f} seconds)"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER inference with HuggingFace models.")
    parser.add_argument(
        "--train_data",
        default="data/finetuning_ner/drug_disease/k_fold_splits/train_fold_4.json",
        help="Path to the training data file (JSON). Default: 'data/finetuning_ner/drug_disease/k_fold_splits/train_fold_4.json'.",
    )
    parser.add_argument(
        "--test_data",
        default="data/finetuning_ner/drug_disease/k_fold_splits/test_fold_4.json",
        help="Path to the test data file (JSON). Default: 'data/finetuning_ner/drug_disease/k_fold_splits/test_fold_4.json'.",
    )
    parser.add_argument(
        "--test_data_input",
        default="data/pubmed_animal_docs/animal_pubmed_docs_part_0_test.csv",
        help="Path to the test data file (CSV). Default: 'data/pubmed_animal_docs/animal_pubmed_docs_part_1.csv'.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/model_predictions/",
        help="Directory to save output annotations. Default: 'models/model_predictions/'.",
    )
    parser.add_argument(
        "--model_name",
        default="michiyasunaga/BioLinkBERT-base",
        help="HuggingFace model name. Default: 'michiyasunaga/BioLinkBERT-base'.",
    )
    parser.add_argument(
        "--model_path",
        default="models/label_all_fixed_full_ds/michiyasunaga_BioLinkBERT-base/epochs_10_data_size_100_iter_1/",
        help="Path to the HuggingFace model directory. Default: 'models/michiyasunaga_BioLinkBERT-base/epochs_15_data_size_100_iter_4/'.",
    )
    parser.add_argument(
    "--output_file_suffix",
    default="label_all_custom_grouping_NEW_##",
    help="Suffix to add to the output file name for unique identification.",
    )
    parser.add_argument(
        "--bio_format",
        action="store_true",
        help="If set, save annotations in BIO format. Default is tuple format.",
    )
    parser.add_argument(
        "--text_column",
        default="Text",
        help="Name of the text column in the CSV file. Default is 'Text'.",
    )
    parser.add_argument(
        "--use_custom_entity_grouping",
        action="store_true",
        help="If set, use custom entity grouping during inference.",
    )
    
    args = parser.parse_args()

    try:
        # Validate paths
        #train_data_path = validate_path(args.train_data)
        test_data_path = validate_path(args.test_data_input)
        output_dir = validate_path(args.output_dir, is_directory=True)
        model_path = validate_path(args.model_path, is_directory=True)

        # Run the inference
        run_inference(
            model_name=args.model_name,
            model_path=model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            output_file_suffix=args.output_file_suffix,
            text_column=args.text_column,
            use_bio_format=args.bio_format,
            custom_entity_grouping=args.use_custom_entity_grouping,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
