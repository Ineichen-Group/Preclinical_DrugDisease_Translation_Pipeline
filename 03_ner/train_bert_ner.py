import os
import argparse
import wandb
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def tokenize_and_align_labels(examples, tokenizer, text_column_name, label_column_name, label_to_id,
                              label_all_tokens=False, padding=True, max_length=512):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=max_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    word_ids_list = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids_list.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                current_label = label[word_idx]
                # Test IO annotation schema
                current_label_to_inside_label = current_label.replace("B-", "I-")
                label_ids.append(label_to_id[current_label_to_inside_label])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                current_label = label[word_idx]
                # Subwords should be annotated as INSIDE the enity, e.g, 'ni' + '##ema' + '##nn - pick type c'
                current_label_to_inside_label = current_label.replace("B-", "I-")
                label_ids.append(label_to_id[current_label_to_inside_label] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_list
    return tokenized_inputs


# Define your function for computing metrics here
def compute_metrics(p, label_list):
    metric = load_metric("seqeval")
    return_entity_level_metrics = False
    return_macro_metrics = False

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # This is just flattening the result dict
        # e.g. {'MISC': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'number': 1}, 'PER': {'precision': 1.0, 'recall': 0.5, 'f1': 0.66, 'number': 2}, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # -> {'MISC_precision': 0.0, 'MISC_recall': 0.0, 'MISC_f1': 0.0, 'MISC_number': 1, 'PER_precision': 1.0, 'PER_recall': 0.5, 'PER_f1': 0.66, 'PER_number': 2, 'overall_precision': 0.5, 'overall_recall': 0.33, 'overall_f1': 0.4, 'overall_accuracy': 0.66}
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    if return_macro_metrics:
        Ps, Rs, Fs = [], [], []
        for type_name in results:
            if type_name.startswith("overall"):
                continue
            print('type_name', type_name)
            Ps.append(results[type_name]["precision"])
            Rs.append(results[type_name]["recall"])
            Fs.append(results[type_name]["f1"])
        return {
            "macro_precision": np.mean(Ps),
            "macro_recall": np.mean(Rs),
            "macro_f1": np.mean(Fs),
        }
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def setup_logging():
    """Configure the logger."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a NER model.")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--percentage", type=int, default=100, help="Training data percentage to use")
    parser.add_argument("--i", type=int, default=1, help="Iteration number")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--val_data_path", type=str, help="Path to the validation dataset")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_name_or_path", type=str, default='michiyasunaga/BioLinkBERT-base', help="HuggingFace model path or name")
    return parser.parse_args()


def prepare_dataset(tokenizer, raw_datasets, label_to_id, max_length=512, label_all_tokens=True, padding=True):
    """Tokenize and align labels across datasets."""
    map_args = {
        "tokenizer": tokenizer,
        "text_column_name": "tokens",
        "label_column_name": "ner_tags",
        "label_to_id": label_to_id,
        "label_all_tokens": label_all_tokens,
        "padding": padding,
        "max_length": max_length,
    }

    datasets = {}
    for split in raw_datasets:
        datasets[split] = raw_datasets[split].map(
            tokenize_and_align_labels,
            batched=True,
            desc=f"Tokenizing {split} dataset",
            fn_kwargs=map_args,
        )
    return datasets


def main():
    setup_logging()
    args = parse_args()

    logger.info("Starting run with arguments: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir="data/huggingface_cache")

    # Load datasets
    data_files = {"train": args.train_data_path, "test": args.test_data_path}
    if args.val_data_path:
        data_files["validation"] = args.val_data_path

    raw_datasets = load_dataset("json", data_files=data_files)
    label_list = get_label_list(raw_datasets["train"]["ner_tags"])
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    # Tokenize and preprocess
    datasets = prepare_dataset(tokenizer, raw_datasets, label_to_id)
    train_dataset = datasets["train"]
    valid_dataset = datasets.get("validation")
    test_dataset = datasets["test"]

    # Sample training data
    num_samples = int(len(train_dataset) * args.percentage / 100)
    np.random.seed()
    sampled_subset = train_dataset.select(np.random.choice(len(train_dataset), size=num_samples, replace=False))
    logger.info(f"Using {len(sampled_subset)} samples from training set")

    # Build model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        label2id=label_to_id,
        id2label=id_to_label,
        cache_dir="data/huggingface_cache"
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir="data/huggingface_cache"
    )

    # Configure training
    suffix = f"epochs_{args.n_epochs}_data_size_{args.percentage}_iter_{args.i}"
    model_name_str = args.model_name_or_path.replace("/", "_")
    run_name = f"{model_name_str}_{suffix}"

    training_args = TrainingArguments(
        output_dir=f"{args.output_path}/results/{model_name_str}/{suffix}",
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{args.output_path}/logs/{suffix}",
        logging_strategy="epoch",
        evaluation_strategy="epoch" if valid_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(valid_dataset),
        metric_for_best_model="eval_loss" if valid_dataset else None,
        greater_is_better=False,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
    )

    # Initialize WandB
    wandb.init(project="preclinical-ner", group="strain", name=run_name, config=training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sampled_subset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_list)
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if valid_dataset:
        logger.info("Evaluating model...")
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(valid_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Running prediction...")
    test_results = trainer.predict(test_dataset, metric_key_prefix="test")
    test_metrics = test_results.metrics
    test_metrics["test_samples"] = len(test_dataset)
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    wandb.finish()
    logger.info("Run complete.")

if __name__ == "__main__":
    main()