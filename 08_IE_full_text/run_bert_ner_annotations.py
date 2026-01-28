import os
import sys
import datetime
import argparse
import time

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


def run_inference(
    model_name,
    model_path,
    test_data_csv_path,
    output_dir,
    output_file_suffix,
    text_column="Text",
    train_data_path=None,
    test_data_path=None,
    use_bio_format=True,
    custom_entity_grouping=False):
    """
    Run inference using a HuggingFace model and save annotations.

    Args:
        model_name (str): Name of the HuggingFace model.
        model_path (str): Path to the model directory.
        train_data_path (str): Path to the training data JSON.
        test_data_path (str): Path to the test data JSON.
        test_data_csv_path (str): Path to the test data CSV.
        output_dir (str): Directory to save output annotations.
        use_bio_format (bool): Whether to save output in BIO format.
        custom_entity_grouping (bool): Whether to use custom entity grouping.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Starting inference...")
    print(f"Model name: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Test CSV path: {test_data_csv_path}")
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
    else:
        print(f"Running inference in tuple format for {model_name}...")
        predictions = model.annotate(test_data_csv_path, source_column=text_column)
    predictions = predictions.drop(columns=[text_column])
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
        #test_data_csv = "/Users/sdoneva/Documents/Work/In Progress/PhD/PhD Projects/In Progress/PreclinicalInfoExtraction/models/model_predictions/bert_ner_full_ds/debug_tokenizer/preds_tokenizer_error_0.csv"# args.test_data_csv
        #output_dir = "/Users/sdoneva/Documents/Work/In Progress/PhD/PhD Projects/In Progress/PreclinicalInfoExtraction/models/model_predictions/bert_ner_full_ds/debug_tokenizer/"#args.output_dir
        
        #test_data_csv_path = validate_path(test_data_csv)
        output_dir = validate_path(args.output_dir, is_directory=True)
        model_path = validate_path(args.model_path, is_directory=True)

        # Run the inference
        run_inference(
            model_name=args.model_name,
            model_path=model_path,
            test_data_csv_path=test_data_path,
            output_dir=output_dir,
            output_file_suffix=args.output_file_suffix,
            text_column=args.text_column,
            use_bio_format=args.bio_format,
            custom_entity_grouping=args.use_custom_entity_grouping,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
