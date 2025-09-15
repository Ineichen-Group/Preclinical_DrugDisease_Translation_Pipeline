# FIXED: robust list parsing + Arrow-friendly parquet writes

from pathlib import Path
import ast, json
import pandas as pd

# Use Arrow-backed dtypes in pandas (optional but recommended)
#pd.options.mode.dtype_backend = "pyarrow"

# Paths
base = Path("06_preclin_clinic_join/data/joined_data")
path_linked = base / "condition_clinical_and_preclinical_15250.csv"
path_clin   = base / "clinical_metadata_mapped.csv"
path_pre    = base / "preclinical_metadata_mapped_annotated_20250723_norm_strain_country.csv"

# --- helpers ---------------------------------------------------------------

def parse_listlike(x):
    """Return a Python list from messy CSV list-looking strings.
    Handles NaN, '', already-lists, '[...]' with single/double quotes,
    and even quoted whole-strings like '"[...]"'. Never uses eval().
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s == "[]":
        return []
    # unwrap if the whole thing is quoted: '"[...]"' or "'[...]'"
    if len(s) > 2 and s[0] in "\"'" and s[-1] == s[0] and s[1] == "[" and s[-2] == "]":
        s = s[1:-1]
    # try Python literal first (handles single quotes)
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else [out]
    except Exception:
        pass
    # then try JSON (double quotes)
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else [out]
    except Exception:
        # last resort: keep as singleton to avoid data loss
        return [s]

def coerce_list_of_ints(lst):
    """Make best-effort to coerce a list of strings to ints; ignore failures."""
    out = []
    for v in lst:
        try:
            out.append(int(v))
        except Exception:
            # keep as-is if not coercible
            out.append(v)
    return out

def normalize_text_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.normalize("NFKC")
                .str.strip()
                .str.lower()
            )
    return df

# --- Load CSVs -------------------------------------------------------------

mapped_studies = pd.read_csv(path_linked)
df_clinical    = pd.read_csv(path_clin)
df_preclinical = pd.read_csv(path_pre)

# --- Parse list-like columns (no eval) ------------------------------------

# Adjust this list to all list-like columns present in each file
mapped_list_cols = ["clinical_doc_ids", "preclinical_doc_ids", "phase", "overall_status"]
for col in mapped_list_cols:
    if col in mapped_studies.columns:
        mapped_studies[col] = mapped_studies[col].apply(parse_listlike)

# If preclinical_doc_ids should be integers:
if "preclinical_doc_ids" in mapped_studies.columns:
    mapped_studies["preclinical_doc_ids"] = mapped_studies["preclinical_doc_ids"].apply(coerce_list_of_ints)

# If clinical/preclinical CSVs also contain list-like columns, add them here:
clin_list_cols = []  # e.g., ["mesh_terms", "conditions"]
for col in clin_list_cols:
    if col in df_clinical.columns:
        df_clinical[col] = df_clinical[col].apply(parse_listlike)

pre_list_cols = []   # e.g., ["pmids", "species_list"]
for col in pre_list_cols:
    if col in df_preclinical.columns:
        df_preclinical[col] = df_preclinical[col].apply(parse_listlike)

# --- Normalize text once (only if these columns exist) --------------------

mapped_studies = normalize_text_columns(mapped_studies, ["disease", "drug", "country", "first_author"])
df_clinical    = normalize_text_columns(df_clinical,    ["disease", "drug", "country", "first_author"])
df_preclinical = normalize_text_columns(df_preclinical, ["disease", "drug", "country", "first_author"])

# --- Save to Parquet (Arrow) ----------------------------------------------

mapped_studies.to_parquet(
    base / "condition_clinical_and_preclinical_15250.parquet",
    engine="pyarrow",
    index=False,
    compression="zstd",
)

df_clinical.to_parquet(
    base / "clinical_metadata_mapped.parquet",
    engine="pyarrow",
    index=False,
    compression="zstd",
)

df_preclinical.to_parquet(
    base / "preclinical_metadata_mapped_annotated_20250723_norm_strain_country.parquet",
    engine="pyarrow",
    index=False,
    compression="zstd",
)
