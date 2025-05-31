#!/usr/bin/env python3
"""
extract_sentences.py

A script that:
1. Reads raw text from a specified column in an input CSV.
2. Splits text into “preliminary” sentences using NLTK’s Punkt tokenizer.
3. Re-tokenizes each preliminary sentence into word/punctuation tokens.
4. Merges consecutive sentences if they match “no-split” patterns (using should_merge).
5. Writes out a CSV where each row corresponds to one (possibly merged) sentence.

Usage:
    python extract_sentences.py \
        --input_csv ./data/your_input_file.csv \
        --text_col Text \
        --output_csv ./data/sentence_split_merged.csv
"""

import argparse
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download Punkt models if not already present. Comment out after first run if desired.
nltk.download("punkt", quiet=True)


def should_merge(prev, next_):
    """
    Determines whether to merge the current sentence‐ending token
    with the next token, based on citation patterns, abbreviations,
    numbers, and other “no‐split” cases in scientific writing.
    """
    # 1. Exact lowercase matches for common medical abbreviations:
    if prev.strip().lower() in [
        'i.c.v', 'i.p', 'i.v', 'i.m', 's.c', 'i.t', 'p.o',
        'inh', 'no', 'al', 'fig', 'figs', 'table', 'ref'
    ]:
        return True

    # 2. If both tokens are entirely lowercase (e.g., continuation),
    #    merge immediately.
    if prev.islower() and next_.islower():
        return True

    # 3. PREVIOUS‐TOKEN patterns (avoid splitting after certain abbreviations)
    prev_patterns = [
        # Medical‐abbrev like "i.p.", "i.v.", etc.
        r'^(?:i\.c\.v|i\.p|i\.v|i\.m|s\.c|i\.t|p\.o)\.$',

        # Fig., Figs., Table., Ref., Inc., etc.
        r'\b(?:Fig|Figs|Table|Ref|Inc|inc|c|p|g|no|No|vs|sp|spp)\.$',

        # et al.
        r'et al\.$',

        # Single uppercase initial, e.g. “A.” or “.A.”
        r'(?:\s|\.)[A-Z]\.$',

        # Citations ending with “12.34)”
        r'\d+\.\d+\)$',

        # ==== NEW: any sequence of uppercase letters separated by dots, ending in a dot ====
        # e.g. "S.D.", "U.S.A.", "N.A.S.A." will match here.
        r'^[A-Z](?:\.[A-Z])*\.$',
    ]

    # 4. NEXT‐TOKEN patterns (if next token fits these, we should NOT split)
    next_patterns = [
        r'^,$',
        r'^\)$',

        # Lowercase‐start of any length, e.g. "of", "into", "protein", "etc."
        r'^[a-z].*',

        r'^\(',         # opening parenthesis
        r'^Mb\b',       # the token "Mb"
        r'^M\b',        # the token "M"
        r'^\d',         # tokens beginning with a digit
        r'^al$',        # "al"
        r'^al\.$',      # "al."
        r'^et$',        # "et"
        r'^\w+\,?\s?\d{4}[a-z]?$',  # citation year like "2000b"
        r'^\r\n$',      # a raw newline
    ]

    # Check each prev_pattern against the entire prev token
    for pattern in prev_patterns:
        if re.fullmatch(pattern, prev):
            return True

    # Check each next_pattern against the entire next token
    for pattern in next_patterns:
        if re.fullmatch(pattern, next_):
            return True

    # If none of the rules fired, do NOT merge
    return False


def split_with_nltk(raw_text):
    """
    Use NLTK’s pretrained Punkt sentence tokenizer to split raw_text into
    a list of (sentence_string, token_list) tuples.
    """
    sentences = sent_tokenize(raw_text)
    out = []
    for sent in sentences:
        toks = word_tokenize(sent)
        out.append((sent, toks))
    return out


def merge_sentences(prelim_sentences):
    """
    Take a list of (sentence_string, token_list) pairs and merge
    adjacent pairs if should_merge says “don’t split here.”

    When the previous sentence ends in a standalone ".", but the token before
    it is something like "S.D", we build "S.D." as prev_candidate so that
    the abbreviation check can catch it.
    """
    merged = []

    for sent_str, toks in prelim_sentences:
        if not merged:
            # First sentence → no merging yet
            merged.append((sent_str, toks))
            continue

        # Look at the last (already‐merged) sentence
        prev_str, prev_toks = merged[-1]

        # Build prev_candidate:
        # If last token is ".", but there is a token before it,
        # combine prev_toks[-2] + "." so that "i.p." or "S.D." is visible
        if prev_toks and prev_toks[-1] == "." and len(prev_toks) >= 2:
            prev_candidate = prev_toks[-2] + "."
        else:
            prev_candidate = prev_toks[-1] if prev_toks else ""

        # The next sentence’s first token:
        next_first_tok = toks[0] if toks else ""

        # Decide if we merge or not
        if should_merge(prev_candidate, next_first_tok):
            # Merge the two sentences
            new_toks = prev_toks + toks
            new_str = prev_str.rstrip() + " " + sent_str.lstrip()
            merged[-1] = (new_str, new_toks)
        else:
            merged.append((sent_str, toks))

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize raw text, split into sentences with NLTK, "
                    "merge based on custom rules (including i.p., S.D., etc.), "
                    "and save to CSV."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--text_col",
        type=str,
        required=True,
        help="Name of the column in the input CSV that contains the raw text to split."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path where the output CSV (sentence‐split, merged) will be written."
    )

    args = parser.parse_args()
    input_path = args.input_csv
    text_column = args.text_col
    output_path = args.output_csv

    # Load the input DataFrame
    df = pd.read_csv(input_path)

    # Ensure the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input CSV.")

    new_rows = []

    for _, row in df.iterrows():
        doc_id = row.get("PMID", None)   # If no PMID column, doc_id will be None
        raw_text = row[text_column]

        # 1. Split into preliminary sentences + tokenize each one
        prelim = split_with_nltk(raw_text)

        # 2. Merge adjacent sentences based on should_merge logic
        merged = merge_sentences(prelim)

        # 3. For each merged sentence, build a row with doc_id, sentence_id, tokens, sent_txt
        for sent_id, (sent_str, sent_tokens) in enumerate(merged):
            new_rows.append({
                "doc_id": doc_id,
                "sentence_id": sent_id,
                "tokens": sent_tokens,
                "sent_txt": sent_str
            })

    # Build a DataFrame from the collected sentence‐rows
    sentence_df = pd.DataFrame(new_rows)

    # Write out to the specified output CSV
    sentence_df.to_csv(output_path, index=False)
    print(f"Sentence‐split (with merging fixed) CSV written to: {output_path}")


if __name__ == "__main__":
    main()
