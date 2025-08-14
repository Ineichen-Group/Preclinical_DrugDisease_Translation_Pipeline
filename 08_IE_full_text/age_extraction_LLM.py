#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from num2words import num2words
from tqdm import tqdm
from unsloth import FastLanguageModel


# --------------------------- Prompts ---------------------------

INSTRUCTIONS_DEFAULT = """### TASK ###

EXTRACT THE AGE OF ANIMALS MENTIONED IN THE TEXT.  
RETURN ONLY THE AGE IN A STANDARDIZED FORMAT.  
IF NO AGE IS GIVEN, RETURN `"AGE NOT SPECIFIED"`.

---

### HOW TO THINK (CHAIN OF THOUGHTS) ###

1. READ the sentence carefully.
2. FIND any age-related phrases (e.g., "8 weeks", "P30", "adult", "3 months").
3. LINK each age phrase to the animal(s) it describes.
4. STANDARDIZE the format:  
   - Use `<number> <unit>` (e.g., `8 weeks`, `3 months`)  
   - Keep terms like `adult`, `juvenile`, `neonatal` unchanged  
5. IF no age is mentioned, write: `"AGE NOT SPECIFIED"`

---

### OUTPUT FORMAT ###

- `AGE: <standardized age or descriptor>`

---

### EXAMPLES ###

#### INPUT 1 ####  
Gene deletion was induced in male and female 12- to 20-week-old mice.  
#### OUTPUT 1 ####  
AGE: 12-20 weeks

#### INPUT 2 ####  
Six adult male WAG/Rij rats were used.  
#### OUTPUT 2 ####  
AGE: adult

#### INPUT 3 ####  
Juvenile pigs (approximately 3 months old) were used.  
#### OUTPUT 3 ####  
AGE: 3 months

#### INPUT 4 ####  
For Experiment 2, male young (3-months-old) and aged (23-months-old) rats were used.  
#### OUTPUT 4 ####  
AGE: 3 months, 23 months

#### INPUT 5 ####  
Twenty Sprague Dawley rats were used; no details were provided on age.  
#### OUTPUT 4 ####  
AGE: AGE NOT SPECIFIED

---

### WHAT NOT TO DO ###

- DO NOT include the whole sentence in the output  
- DO NOT include weight, sex, or strain  
- DO NOT guess the age  
- DO NOT omit the unit (e.g., weeks/months)  
- DO NOT ignore terms like "adult", "neonatal", or "juvenile"  
- DO NOT return multiple values or unformatted strings

---
ONLY OUTPUT THE AGE USING THE FORMAT ABOVE. NOTHING ELSE.
"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# --------------------------- Utilities ---------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path, verbose: bool = True):
    ensure_parent_dir(log_path)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout) if verbose else logging.FileHandler(log_path)],
    )
    logging.info("Logging to %s", log_path)


def parse_dtype(dtype_str: str):
    s = (dtype_str or "auto").lower()
    if s in ("auto", "none"):
        return None
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unrecognized dtype: {dtype_str}")


def load_instructions(args) -> str:
    if args.instructions_file:
        return Path(args.instructions_file).read_text(encoding="utf-8")
    return INSTRUCTIONS_DEFAULT


# --------------------------- Model I/O ---------------------------

def load_unsloth_model(model_dir: str,
                       max_seq_length: int,
                       dtype,
                       load_in_4bit: bool):
    logging.info("Loading Unsloth model from: %s", model_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        pass  # Unsloth handles device internally; ignore if to() is unnecessary
    logging.info("CUDA available: %s | visible devices: %s", torch.cuda.is_available(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        logging.info("GPU count: %d | GPU 0: %s", torch.cuda.device_count(), torch.cuda.get_device_name(0))
    return model, tokenizer, device


def format_prompt(instructions: str, text: str) -> str:
    return ALPACA_PROMPT.format(instructions, text, "")


def parse_llm_response(text: str) -> str:
    """
    Robustly extract the response after '### Response:' or 'Response:'.
    """
    # Try the canonical header
    parts = re.split(r"###\s*Response:\s*", text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        resp = parts[1]
    else:
        # Fallback: split on 'Response:'
        parts = text.split("Response:", 1)
        resp = parts[1] if len(parts) == 2 else text
    resp = resp.replace("<|end_of_text|>", "").strip()
    return resp


def extract_age_unsloth(single_text: str,
                        model,
                        tokenizer,
                        instructions: str,
                        device: str,
                        max_new_tokens: int) -> str:
    inputs = tokenizer([format_prompt(instructions, single_text)],
                       return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    decoded = tokenizer.batch_decode(outputs)[0]
    return parse_llm_response(decoded)


# --------------------------- Validation ---------------------------

def is_valid_age_prediction(prediction: str, original_text: str) -> bool:
    if not prediction:
        return False

    standalone_labels = {
        "young adult", "adult", "juvenile", "neonatal",
        "aged", "young", "old", "newborn"
    }

    entries = [entry.strip() for entry in prediction.split(",") if entry.strip()]
    if not entries:
        return False

    text_lower = original_text.lower()

    for entry in entries:
        entry = entry.lower()
        words = entry.split()

        # Allow single-word predictions like "adult"
        if len(words) == 1 and words[0] in standalone_labels:
            continue

        if len(words) != 2:
            return False

        number_part, unit = words

        if not re.fullmatch(r"(day|days|week|weeks|month|months|year|years)", unit, re.IGNORECASE):
            return False

        # Handle range like "12-20" or "17–19"
        range_match = re.match(r"(\d+)[–-](\d+)", number_part)
        if range_match:
            number = range_match.group(2)
        else:
            number = number_part

        digit_form = number
        try:
            word_form = num2words(int(float(number)))
        except Exception:
            word_form = number

        if (digit_form not in text_lower) and (word_form not in text_lower):
            return False

    return True


# --------------------------- Pipeline ---------------------------

def read_input_csv(path: Path,
                   text_col: str,
                   pmid_col: str,
                   sentence_id_col: str,
                   filter_col: str | None,
                   filter_eq: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (text_col, pmid_col, sentence_id_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}")
    if filter_col:
        if filter_col not in df.columns:
            raise ValueError(f"Filter column '{filter_col}' not found in {path}")
        if filter_eq is None:
            raise ValueError("--filter-eq must be provided when --filter-col is used")
        df = df[df[filter_col].astype(str) == str(filter_eq)]
    df = df.copy()
    df["doc_id_unique"] = df[pmid_col].astype(str) + "_" + df[sentence_id_col].astype(str).str.strip()
    return df


def run_predictions(df: pd.DataFrame,
                    text_col: str,
                    pmid_col: str,
                    docid_col: str,
                    model,
                    tokenizer,
                    instructions: str,
                    device: str,
                    out_csv: Path,
                    max_attempts: int,
                    max_new_tokens: int,
                    log_every: int = 50) -> pd.DataFrame:
    ensure_parent_dir(out_csv)
    results = []
    wrote_header = False
    start = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        pmid = row[pmid_col]
        docid = row[docid_col]
        text = row[text_col]

        try:
            attempts = []
            final_pred = None
            for attempt in range(1, max_attempts + 1):
                pred = extract_age_unsloth(text, model, tokenizer, instructions, device, max_new_tokens)
                cleaned = pred.replace("AGE:", "").strip()
                attempts.append(pred)
                if is_valid_age_prediction(cleaned, text):
                    final_pred = pred
                    break
                logging.info("[PMID %s][Attempt %d] Invalid prediction: %s | text: %s", pmid, attempt, pred, text)
            if final_pred is None:
                final_pred = max(attempts, key=lambda p: len(p)) if attempts else "ERROR: No valid attempts"
        except Exception as e:
            final_pred = f"ERROR: {e}"

        logging.info("[PMID %s] Final prediction: %s", pmid, final_pred)

        item = {
            "PMID": pmid,
            "doc_id_unique": docid,
            "sent_txt": text,
            "age_prediction": final_pred,
        }
        results.append(item)

        # Stream to CSV (append)
        df_row = pd.DataFrame([item])
        df_row.to_csv(out_csv, mode="a", index=False, header=not wrote_header)
        wrote_header = True

        if (i + 1) % log_every == 0:
            elapsed = time.time() - start
            logging.info("Processed %d/%d rows in %.1fs (%.1f rows/min)",
                         i + 1, len(df), elapsed, (i + 1) / max(elapsed / 60.0, 1e-6))

    return pd.DataFrame(results)


# --------------------------- CLI ---------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Extract animal age from sentences using an Unsloth LLM fine-tuned with Alpaca-style prompting."
    )
    # I/O
    p.add_argument("--input-csv", required=True, help="Path to input CSV.")
    p.add_argument("--output-csv", required=True, help="Path to write predictions CSV.")
    p.add_argument("--log-path", default=f"server_logs/llm_age_prediction_{now_stamp()}.log",
                   help="Path to write log file.")

    # Columns
    p.add_argument("--text-col", default="sent_txt", help="Column with sentence text.")
    p.add_argument("--pmid-col", default="PMID", help="PMID column.")
    p.add_argument("--sentence-id-col", default="sentence_id", help="Sentence ID column (for doc_id_unique).")

    # Optional pre-filter (e.g., keep only predicted_label==1)
    p.add_argument("--filter-col", default=None, help="Column to filter by equality.")
    p.add_argument("--filter-eq", default=None, help="Value used with --filter-col for equality filtering.")

    # Model / generation
    p.add_argument("--model-dir", required=True,
                   help="Unsloth model path or HF id, e.g. /data/.../lora_model_x or 'unsloth/Meta-Llama-3.1-8B'.")
    p.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length for Unsloth.")
    p.add_argument("--dtype", default="auto",
                   help="Model dtype: auto | float16 | bfloat16 | float32.")
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (default: enabled).")
    p.add_argument("--max-new-tokens", type=int, default=500, help="Max new tokens for generation.")
    p.add_argument("--max-attempts", type=int, default=3, help="Retries for validation before giving up.")

    # Instructions
    p.add_argument("--instructions-file", default=None,
                   help="Path to a file with instructions. If not set, a built-in prompt is used.")

    return p


def main():
    args = build_arg_parser().parse_args()

    # Paths
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    log_path = Path(args.log_path)

    # Logging
    setup_logger(log_path, verbose=True)

    # Instructions
    instructions = load_instructions(args)

    # Dtype + 4bit
    dtype = parse_dtype(args.dtype)
    load_in_4bit = not args.no_4bit

    # Load model
    model, tokenizer, device = load_unsloth_model(
        model_dir=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Read input
    df = read_input_csv(
        path=input_csv,
        text_col=args.text_col,
        pmid_col=args.pmid_col,
        sentence_id_col=args.sentence_id_col,
        filter_col=args.filter_col,
        filter_eq=args.filter_eq,
    )
    logging.info("Loaded %d rows from %s", len(df), input_csv)

    # Prepare/clear output
    ensure_parent_dir(output_csv)
    if output_csv.exists():
        logging.info("Output file exists. Overwriting: %s", output_csv)
        output_csv.unlink(missing_ok=True)

    # Run
    start_time = time.time()
    results_df = run_predictions(
        df=df,
        text_col=args.text_col,
        pmid_col=args.pmid_col,
        docid_col="doc_id_unique",
        model=model,
        tokenizer=tokenizer,
        instructions=instructions,
        device=device,
        out_csv=output_csv,
        max_attempts=args.max_attempts,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - start_time

    logging.info("Finished. Wrote %d rows → %s", len(results_df), output_csv)
    print(f"\nCompleted predictions in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
