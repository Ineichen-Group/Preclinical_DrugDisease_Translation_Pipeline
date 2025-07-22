#!/usr/bin/env python
import re
import os
import gc
import pandas as pd
import argparse

# ——— Regex patterns precompiled for speed —————————————————————————

# Matches trailing time units
UNIT_PATTERN = re.compile(r'\b(weeks?|months?|days?|years?)\b$', re.IGNORECASE)

# Matches weight entries to drop
WEIGHT_PATTERN = re.compile(
    r'\b\d*\.?\d+\s*[-–]?\s*\d*\s*'
    r'(g|gram|grams|kg|kilogram|kilograms)\b',
    re.IGNORECASE
)

# Matches pure “NOT AGE”
NOT_AGE_PATTERN = re.compile(r'^NOT AGE$', re.IGNORECASE)


# ——— Sentence→document combining ————————————————————————————————

def combine_age_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    From sentence-level predictions in `df`, produce one row per document:
    - doc_base: strip off the last underscore segment
    - Extract AGE:… text
    - Clean and dedupe within each doc
    - Collect supporting sentence IDs
    """
    # Derive document base
    df['doc_base'] = df['doc_id_unique'].str.rsplit('_', n=1).str[0]

    # Extract text after “AGE: ”
    df['age_extracted'] = df['age_prediction'] \
        .str.extract(r'AGE:\s*(.*?)(?=\.\s|$)', expand=False)

    # Simple cleanup: split on commas, strip, dedupe+sort
    def clean_age_string(x):
        if pd.isna(x):
            return None
        parts = [p.strip() for p in x.split(',') if p.strip()]
        return ', '.join(sorted(set(parts))) if parts else None

    df['age_prediction_clean'] = df['age_extracted'].apply(clean_age_string)

    # Rows with at least one valid age
    valid = df[df['age_prediction_clean'].notna()]

    # Aggregate per doc_base
    grouped = (
        df.groupby('doc_base')['age_prediction_clean']
          .apply(lambda s: ', '.join(sorted(set(s.dropna()))))
          .reset_index()
          .rename(columns={
              'doc_base': 'doc_id_unique',
              'age_prediction_clean': 'age_prediction'
          })
    )

    # Collect supporting sentence IDs
    supporting = (
        valid.groupby('doc_base')['doc_id_unique']
             .apply(lambda ids: '; '.join(sorted(set(ids))))
             .reset_index()
             .rename(columns={
                 'doc_base': 'doc_id_unique',
                 'doc_id_unique': 'supporting_sentence_ids'
             })
    )

    # Merge them
    combined = pd.merge(grouped, supporting, on='doc_id_unique', how='left')
    combined = combined[combined['doc_id_unique'] != 'doc_id']  # drop header artifacts
    return combined


# ——— Clean “NOT AGE” rows —————————————————————————————————————

def clean_not_age(val: str) -> str:
    if pd.isna(val):
        return val
    parts = [p.strip() for p in val.split(',')]
    # Only “NOT AGE” → “AGE NOT SPECIFIED”
    if parts == ['NOT AGE']:
        return 'AGE NOT SPECIFIED'
    # Otherwise drop any “NOT AGE” tokens
    parts = [p for p in parts if p != 'NOT AGE']
    return ', '.join(parts)


# ——— Normalize & encode age expressions ————————————————————————

def clean_prediction(text: str) -> str:
    """
    Standardizes age expressions, e.g.:
      P10-P17 days   → 10-17 days
      3 or 6 months  → 3 months, 6 months
      14, 21 days    → 14 days, 21 days
    """
    if pd.isna(text):
        return 'age not specified'

    # 1) Normalize whitespace & strip leading “P”
    txt = text.replace('\n', ' ').replace('\t', ' ').strip()
    txt = re.sub(r'\bP(?=\d)', '', txt)

    # 2) Extract trailing unit for later
    unit_m = UNIT_PATTERN.search(txt)
    default_unit = unit_m.group(1).lower() if unit_m else None

    # 3) Expand “14, 21, 56 days” → “14 days, 21 days, 56 days”
    m = re.match(
        r'^(?P<numlist>\d+(?:\s*,\s*\d+)*)\s*'
        r'(?P<unit>weeks?|months?|days?|years?)$',
        txt, re.IGNORECASE
    )
    if m:
        nums = [n.strip() for n in m.group('numlist').split(',')]
        txt = ', '.join(f"{n} {m.group('unit')}" for n in nums)

    # 4) “and”/“or” → commas
    txt = re.sub(r'\s+and\s+', ' , ', txt, flags=re.IGNORECASE)
    txt = re.sub(r'\s+or\s+', ' , ', txt, flags=re.IGNORECASE)

    # 5) Ranges “X to Y” or “X weeks to Y weeks” → “X-Y unit”
    txt = re.sub(
        r'(\d+)\s*(weeks?|months?|days?|years?)\s+to\s+'
        r'(\d+)\s*(weeks?|months?|days?|years?)',
        lambda mm: f"{mm.group(1)}-{mm.group(3)} {mm.group(2)}",
        txt, flags=re.IGNORECASE
    )
    txt = re.sub(r'(\d+)\s+to\s+(\d+)', r'\1-\2', txt, flags=re.IGNORECASE)
    txt = re.sub(r'\s*[-–]\s*', '-', txt)

    # 6) Split out any AGE:… segments, else full text
    parts = []
    segs = re.findall(
        r'AGE:\s*(.*?)(?=\s*(AGE:|$|###|---))',
        txt, flags=re.IGNORECASE
    )
    if segs:
        for seg, _ in segs:
            parts.extend(p.strip() for p in seg.split(',') if p.strip())
    else:
        parts.extend(p.strip() for p in txt.split(',') if p.strip())

    # 7) Append default unit to bare numbers/ranges
    if default_unit:
        for i, p in enumerate(parts):
            if re.match(r'^\d+(?:-\d+)?$', p):
                parts[i] = f"{p} {default_unit}"

    # 8) Drop weight entries
    parts = [p for p in parts if not WEIGHT_PATTERN.search(p)]

    # 9) Handle nonspecific labels (if no specifics, keep “unknown” etc.)
    nonspec = re.compile(
        r'^(age\s*)?(not specified|unknown|unspecified)$',
        re.IGNORECASE
    )
    specific = [p for p in parts if not nonspec.fullmatch(p)]
    use = specific if specific else [p for p in parts if nonspec.fullmatch(p)]

    # 10) Dedupe in original order
    seen = set()
    final = []
    for p in use:
        if p not in seen:
            seen.add(p)
            final.append(p)

    return ', '.join(final) if final else 'age not specified'


# ——— Main: chunked processing to avoid OOM —————————————————————————

def main(input_csv: str, output_csv: str = None, chunksize: int = 100_000):
    # Derive output path if not provided
    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_doc_level_predictions{ext}"

    first = True
    reader = pd.read_csv(
        input_csv,
        names=['doc_id', 'doc_id_unique', 'ent_text', 'age_prediction'],
        usecols=[1, 3],
        chunksize=chunksize,
        iterator=True,
        low_memory=True
    )

    for chunk in reader:
        # 1) Combine
        combined = combine_age_predictions(chunk)

        # 2) Clean NOT AGE and strip trailing junk
        combined['age_prediction'] = (
            combined['age_prediction']
              .apply(clean_not_age)
              .str.replace(r'###.*', '', regex=True)
              .str.strip()
        )

        # 3) Encode final labels
        combined['prediction_encoded_label'] = \
            combined['age_prediction'].apply(clean_prediction)

        # 4) Rename & write
        combined = combined.rename(columns={'doc_id_unique': 'PMID'})
        combined.to_csv(
            output_csv,
            mode='w' if first else 'a',
            index=False,
            header=first
        )
        first = False

        # 5) Free memory
        del chunk, combined
        gc.collect()

    print(f"Processed and saved to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine & clean age predictions (chunked).'
    )
    parser.add_argument(
        '-i', '--input',
        default='08_IE_full_text/model_predictions/age/age_unsloth_meta_llama_3.1_8b.csv',
        help='Raw predictions CSV'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output CSV (defaults to input + _doc_level_predictions)'
    )
    parser.add_argument(
        '-c', '--chunksize', type=int, default=100_000,
        help='Rows per chunk'
    )
    args = parser.parse_args()
    main(args.input, args.output, args.chunksize)
