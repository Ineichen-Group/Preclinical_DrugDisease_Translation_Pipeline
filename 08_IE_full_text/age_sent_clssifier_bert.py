#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
import sys

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import TrainingArguments
import torch
import psutil

def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    print(f"[MEMORY] {stage}: {mem_mb:.2f} MB")
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a CSV containing columns: PMID, sentence_id, sent_txt. "
                    "Outputs a CSV with an added `predicted_label` column."
    )
    parser.add_argument(
        "model_dir",
        help="Path to the directory (or HuggingFace identifier) containing the fine-tuned model."
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV with columns: PMID, sentence_id, sent_txt"
    )
    parser.add_argument(
        "output_csv",
        help="Path where the output CSV (including the new `predicted_label` column) will be written"
    )
    return parser.parse_args()

def tokenize_function(batch, tokenizer):
    return tokenizer(batch["sent_txt"], truncation=True, padding="max_length", max_length=128)

def main():
    args = parse_args()

    # 1) Verify input file exists
    if not os.path.isfile(args.input_csv):
        print(f"Error: input CSV '{args.input_csv}' not found.", file=sys.stderr)
        sys.exit(1)

    # 2) Load the tokenizer & model
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"Error loading model/tokenizer from '{args.model_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Read the CSV into a DataFrame and check required columns
    df = pd.read_csv(args.input_csv)
    log_memory("After loading CSV")

    required_cols = ["PMID", "sentence_id", "sent_txt"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: column '{col}' not found in input CSV.", file=sys.stderr)
            sys.exit(1)

    # 4) Create a HuggingFace Dataset from the DataFrame
    hf_dataset = Dataset.from_pandas(df, split="train")
    hf_dataset = hf_dataset.map(lambda batch: tokenize_function(batch, tokenizer), batched=True)
    hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    log_memory("After tokenization")

    # 5) Use a Trainer for inference
    training_args = TrainingArguments(
        output_dir="./tmp_predictions",
        per_device_eval_batch_size=8,
        dataloader_drop_last=False,
        dataloader_num_workers=2,
        report_to="none"
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args
        )
    log_memory("Before prediction")

    # 6) Run prediction
    with torch.no_grad():
        predictions = trainer.predict(hf_dataset)
    logits = predictions.predictions
    pred_labels = np.argmax(logits, axis=1)

    # 7) Add `predicted_label` column to the original DataFrame
    df_out = df.copy()
    df_out["predicted_label"] = pred_labels

    # 8) Write output CSV
    try:
        df_out.to_csv(args.output_csv, index=False)
    except Exception as e:
        print(f"Error writing to output CSV '{args.output_csv}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Inference complete. Wrote {len(df_out)} rows (with `predicted_label`) → '{args.output_csv}'")

if __name__ == "__main__":
    main()
