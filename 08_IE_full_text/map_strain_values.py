#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import re
from rapidfuzz import process, fuzz
from tqdm import tqdm
import glob

tqdm.pandas()

mapping_log = []  # global log list

# =========================
# Synonym/fuzzy normalization
# =========================
def build_lookup_table(synonyms_file: str):
    """
    Load synonyms and create a canonical lookup dict (case-insensitive).
    Expected columns: Synonym, StrainName
    """
    df = pd.read_csv(synonyms_file)
    syn_to_can_raw = df.set_index("Synonym")["StrainName"].to_dict()
    canonical_lookup = {str(k).lower(): v for k, v in syn_to_can_raw.items()}
    canonical_lookup.update({str(v).lower(): v for v in syn_to_can_raw.values()})
    return canonical_lookup

def map_strain(s, canonical_lookup, choices, cutoff=93):
    """
    Map a single raw token via exact case-insensitive lookup, then fuzzy fallback.
    """
    if pd.isna(s):
        mapping_log.append((s, s))
        return s

    s_str = str(s).strip()
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

    # strip trailing 'J' the fuzzy step might have introduced if source didn't have it
    if isinstance(result, str) and result.lower().endswith("j") and not s_lower.endswith("j"):
        result = result[:-1]

    mapping_log.append((s_str, result))
    return result

def normalize_animal_strains(
    df, canonical_lookup, column="animal_strain", new_column="animal_strain_norm", delimiter=","
):
    """
    Apply synonym/fuzzy mapping per token (split by delimiter), dedupe in-cell.
    """
    choices = list(canonical_lookup.keys())

    def normalize_entry(entry):
        if pd.isna(entry):
            return entry
        strains = [s.strip() for s in str(entry).split(delimiter)]
        normalized = [map_strain(s, canonical_lookup, choices) for s in strains]
        # keep order, drop empties
        unique_normalized = []
        seen = set()
        for x in normalized:
            if pd.isna(x):
                continue
            x = str(x).strip()
            if not x or x in seen:
                continue
            seen.add(x)
            unique_normalized.append(x)
        return delimiter.join(unique_normalized)

    df[new_column] = df[column].progress_apply(normalize_entry)
    return df

# =========================
# Generic family-level post-processor (NEW, proper casing)
# =========================

# Reserved placeholders to skip normalization
RESERVED_TOKENS = {"NOFULLTEXT", "UNLABELED", "UNKNOWN", "NA", "NONE"}

def _clean_for_matching(token: str) -> str:
    """
    Uppercase matching string and remove noise (used only for regex matching).
    """
    t = token.upper()
    t = re.sub(r'-?TG\([^)]*\).*', '', t)      # drop transgene tails like -Tg(...)
    t = re.sub(r'\([^)]*\)', '', t)            # drop any (...)
    t = re.sub(r"[^A-Z0-9/\s\-_]", " ", t)     # remove weird punctuation
    t = t.replace("\\", "/")
    t = re.sub(r"[\s\-_]+", "", t)             # remove spaces, hyphens, underscores
    t = re.sub(r"/+", "/", t)                  # collapse multiple slashes
    return t

def _clean_preserve_case(token: str) -> str:
    """
    Case-preserving cleanup for fallback output (no uppercasing).
    """
    t = token
    t = re.sub(r'-?Tg\([^)]*\).*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\([^)]*\)', '', t)
    t = re.sub(r"[^A-Za-z0-9/\s\-_]", " ", t)
    t = t.replace("\\", "/")
    t = re.sub(r"[\s\-_]+", "", t)             # collapse spaces/hyphens/underscores
    t = re.sub(r"/+", "/", t)
    return t

def normalize_strain_token_generic(token: str) -> str:
    """
    Normalize a single token to a family-level canonical form with proper casing.
    Handles common mouse and rat families and slash/substring variants.
    """
    if token is None:
        return ""
    orig = str(token).strip()
    if not orig:
        return ""

    # Reserved placeholders → return as-is (preserve casing)
    if orig.upper() in RESERVED_TOKENS:
        return orig

    t_match = _clean_for_matching(orig)        # UPPER, sanitized (for pattern detection)
    t_fallback = _clean_preserve_case(orig)    # case-preserving sanitized (for fallback output)

    # ---- Mouse families ----
    m = re.match(r"^C\s*57\s*[/\-]?\s*B?L?\s*[/\-]?\s*(?P<num>\d+)", t_match)
    if m:
        return f"C57BL/{m.group('num')}"

    m = re.match(r"^(BALB)/?C(\b|$)", t_match)
    if m:
        return "BALB/C"

    m = re.match(r"^DBA/?(?P<num>\d+)", t_match)
    if m:
        return f"DBA/{m.group('num')}"

    m = re.match(r"^(FVB)(?:/?([A-Z]))?", t_match)
    if m:
        return "FVB/N"

    if re.match(r"^129", t_match):
        return "129"

    m = re.match(r"^(CBA)(?:/([A-Z0-9]+))?$", t_match)
    if m:
        return "CBA"

    m = re.match(r"^(NOD)(?:/([A-Z0-9]+))?$", t_match)
    if m:
        return "NOD"

    m = re.match(r"^(C3H)(?:/([A-Z0-9]+))?$", t_match)
    if m:
        return "C3H"

    m = re.match(r"^(SJL)(?:/([A-Z0-9]*))?$", t_match)
    if m:
        return "SJL"

    # ---- Rat / outbred families ----
    if re.match(r"^SPRAGUE(?:[A-Z]{0,6})?[\s\-]*DAWLE?Y", t_match):
        return "Sprague-Dawley"
    if re.match(r"^WISTAR", t_match):
        return "Wistar"
    if re.match(r"^LONG(?:[A-Z]{0,6})?[\s\-]*EVA[NS]{1,2}", t_match):
        return "Long-Evans"
    if re.match(r"^FISCHER?344", t_match) or re.match(r"^F344", t_match):
        return "Fischer"
    if re.match(r"^LEWIS$", t_match) or re.match(r"^LEW(?=$|[^A-Z])", t_match):
        return "Lewis"
    if re.match(r"^SWISSALBINO", t_match):
        return "Swiss Albino"

    # ---- Generic harmonization ----
    # Drop trailing substrain suffix like /J, /N, etc.
    t_out = re.sub(r"/[A-Za-z]+$", "", t_fallback)
    if "/" in t_out:
        a, b, *rest = t_out.split("/")
        if a and b:
            return f"{a}/{b}"

    # Otherwise return cleaned token (case preserved)
    return t_out

def postprocess_family_level(cell: str, delimiter: str) -> str:
    """
    Apply generic family-level normalization to a delimiter-separated cell.
    """
    if pd.isna(cell) or str(cell).strip() == "":
        return ""
    parts = [p.strip() for p in str(cell).split(delimiter) if p.strip() != ""]
    normed = [normalize_strain_token_generic(p) for p in parts]
    # de-dupe while preserving order
    seen, out = set(), []
    for n in normed:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return delimiter.join(out)

def build_postprocess_audit(df: pd.DataFrame, column: str, delimiter: str) -> pd.DataFrame:
    """
    Build an audit mapping of every seen token in `column` -> family-level normalized token.
    """
    tokens = (
        df[column].astype(str)
        .str.split(delimiter)
        .explode()
        .dropna()
        .map(str.strip)
    )
    tokens = tokens[tokens != ""]
    variants = tokens.drop_duplicates()
    mapped = variants.apply(lambda x: (x, normalize_strain_token_generic(x))).tolist()
    audit_df = pd.DataFrame(mapped, columns=["variant", "family_normalized"]).drop_duplicates()
    return audit_df

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Normalize animal strain names using fuzzy matching + generic family post-processing."
    )
    parser.add_argument("--input_csv", help="Path to input CSV with strain column.")
    parser.add_argument("--input_dir", help="Directory containing chunk_*.csv files to merge and process.")
    parser.add_argument("--output_csv", required=True, help="Path to save normalized output.")
    parser.add_argument("--synonyms_file", default="./data/strain_normalization/combined_df_mice_rat.csv",
                        help="Path to synonyms CSV (columns: Synonym, StrainName).")
    parser.add_argument("--column", default="animal_strain", help="Column name in input CSV to normalize.")
    parser.add_argument("--new_column", default="animal_strain_norm", help="Name of column for normalized output.")
    parser.add_argument("--delimiter", default=",", help="Delimiter for multiple strains in one cell.")
    parser.add_argument("--post_column", default=None,
                        help="Name of column for family-level post-processing (default: <new_column>_family).")
    parser.add_argument("--write_audit", default=None, help="Optional path to write variant→family audit CSV.")
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
        print(f"Loaded {len(df)} rows.")
    else:
        raise ValueError("Either --input_csv or --input_dir must be specified.")

    # Load synonym table
    print(f"Loading synonym lookup from: {args.synonyms_file}")
    canonical_lookup = build_lookup_table(args.synonyms_file)

    # Normalize (fuzzy/synonym)
    print(f"Normalizing strain names in column: {args.column}")
    df = normalize_animal_strains(
        df,
        canonical_lookup,
        column=args.column,
        new_column=args.new_column,
        delimiter=args.delimiter,
    )

    # Post-process to family-level generic forms
    post_col = args.post_column or f"{args.new_column}_family"
    print(f"Post-processing to family-level in column: {post_col}")
    df[post_col] = df[args.new_column].progress_apply(lambda x: postprocess_family_level(x, args.delimiter))

    # Optional audit map
    if args.write_audit:
        print(f"Writing variant→family audit to: {args.write_audit}")
        audit_df = build_postprocess_audit(df, args.new_column, args.delimiter)
        os.makedirs(os.path.dirname(args.write_audit), exist_ok=True)
        audit_df.to_csv(args.write_audit, index=False)

    # Save
    print(f"Saving to: {args.output_csv}")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
