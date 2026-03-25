import os
import sys
import datetime
import argparse

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
    test_data_path,
    output_dir,
    output_file_suffix,
    train_data_path=None,
    use_bio_format=True,
    custom_entity_grouping=False):
    """
    Run inference using a HuggingFace model and save annotations.

    Args:
        model_name (str): Name of the HuggingFace model.
        model_path (str): Path to the model directory.
        train_data_path (str): Path to the training data JSON.
        test_data_path (str): Path to the test data JSON or CSV.
        output_dir (str): Directory to save output annotations.
        use_bio_format (bool): Whether to save output in BIO format.
        custom_entity_grouping (bool): Whether to use custom entity grouping.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = NERModel(
        "huggingface",
        model_name,
        model_path,
        use_custom_entities_grouping=custom_entity_grouping,
    )
    model_name_clean = model_name.split("/")[-1]

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    output_file_format = (
    f"test_annotated_{model_name_clean}_{'BIO' if use_bio_format else 'tuples'}_{current_date}{output_file_suffix}.csv"
    )   
    output_path = os.path.join(output_dir, output_file_format)

    # Run inference
    if use_bio_format:
        print(f"Running inference in BIO format for {model_name}...")
        predictions = model.bert_predict_bio_format(
            train_data_path, test_data_path, "tokens", "ner_tags"
        )
        #predictions = predictions.to_pandas()
    else:
        print(f"Running inference in tuple format for {model_name}...")
        predictions = model.annotate(test_data_path, "Text")
    if "Text" in predictions.columns:
        predictions = predictions.drop(columns=['Text'])
    predictions.to_csv(output_path, index=False, sep=",")
    print(f"Annotations saved to {output_path}")


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
        "--test_data_csv",
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
    
    args = parser.parse_args()

    try:
        # Validate paths
        output_dir = validate_path(args.output_dir, is_directory=True)
        model_path = validate_path(args.model_path, is_directory=True)
        
        if args.test_data_csv:
            test_data_path = validate_path(args.test_data_csv)
            train_data_path = None
        else:
            test_data_path = validate_path(args.test_data)
            train_data_path = validate_path(args.train_data)

        # Run the inference
        run_inference(
            model_name=args.model_name,
            model_path=model_path,
            test_data_path=test_data_path,
            output_dir=output_dir,
            output_file_suffix=args.output_file_suffix,
            train_data_path=train_data_path,
            use_bio_format=args.bio_format,
            custom_entity_grouping=True
        )
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
