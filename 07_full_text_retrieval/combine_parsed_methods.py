from pathlib import Path
import pandas as pd
from typing import List, Optional, Set, Union
from cadmus_extractors.utils import load_wrong_pmids
import csv
import re

def process_json_folder(
    folder: Path,
    exclude_pmids: Set[str],
    source_label: str,
    pattern: str = '*.json'
) -> pd.DataFrame:
    """Return one big DataFrame of grouped articles from JSON files."""
    frames = []
    for json_path in folder.glob(pattern):
        df = pd.read_json(json_path)
        if 'doc_id' not in df.columns or 'paragraph' not in df.columns:
            print(f"Skipping {json_path}: missing doc_id or paragraph")
            continue
        df = df.rename(columns={'doc_id': 'pmid'})
        df['pmid'] = df['pmid'].astype(str)
        if df['pmid'].iloc[0] in exclude_pmids:
            continue
        df['subtitle_paragraph'] = (
            df.get('subtitle', '').fillna('') + ' ' +
            df.get('paragraph', '').fillna('')
        )
        merged = (
            df.groupby('pmid')['subtitle_paragraph']
              .apply(' '.join)
              .reset_index(name='Text')
              .rename(columns={'pmid': 'PMID'})
        )
        merged['Source'] = source_label
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['PMID', 'Text', 'Source'])

def normalize_text(text):
    """
    Normalize text for JSONL export via pandas:
    - Remove control characters (except \n and \t)
    - Replace Unicode line/paragraph separators
    - Replace fancy quotes with standard ones
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return text

    # Remove control characters except newline/tab
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # Replace Unicode line/paragraph separators
    text = text.replace("\u2028", " ").replace("\u2029", " ")

    # Replace fancy quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    return text.strip()

def combine_all(
    base_dir: str,
    subfolders: List[str],
    inner_folders: List[Union[None, str, List[str]]],
    exclude_pmids: Set[str],
    output_path: Path
):
    base = Path(base_dir)
    all_frames = []
    folder_counts = {}

    for sub, inner in zip(subfolders, inner_folders):
        base_folder = base / sub

        # Handle single or multiple inner folders
        if inner is None:
            folders_to_process = [base_folder]
        elif isinstance(inner, str):
            folders_to_process = [base_folder / inner]
        else:  # assume list of str
            folders_to_process = [base_folder / i for i in inner]

        total_pmids = 0
        for folder in folders_to_process:
            if not folder.is_dir():
                print(f"Warning: {folder} missing")
                continue

            json_files = list(folder.glob("*.json"))
            if not json_files:
                print(f"Warning: No JSON files found in {folder}")
                continue

            df = process_json_folder(folder, exclude_pmids, source_label=sub)
            count = df['PMID'].nunique()
            total_pmids += count
            all_frames.append(df)

        folder_counts[sub] = total_pmids

    if not all_frames:
        print("No data found in any folder.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    print("\n--- Per-folder document counts ---")
    for sub, cnt in folder_counts.items():
        print(f"{sub:12s}: {cnt}")

    print(f"\nTotal before dedup: {combined['PMID'].nunique()} unique articles")
    combined = combined.drop_duplicates('PMID')
    print(f"Total after dedup : {combined['PMID'].nunique()} unique articles")

    out = output_path / f"combined_materials_methods_{combined['PMID'].nunique()}.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)
    combined["Text"] = combined["Text"].map(normalize_text)

    #combined.to_csv(
      #  out,
       # index=False,
       # quoting=csv.QUOTE_ALL,
       # escapechar="\\"
    #)
    combined.to_json(out, orient="records", lines=True, force_ascii=False)

    print(f"\nSaved combined CSV to {out}, shape: {combined.shape}")


if __name__ == '__main__':
    BASE_DIR = '07_full_text_retrieval/materials_methods'
    SUBFOLDERS = ['html', 'xml', 'plain', 'bioc_json', 'pdf']
    INNER_FOLDERS = [None, None, None, ['MS_methods', 'alzheimer_methods', 'parkinson_methods', 'epilepsy_methods', 'all_pmids_methods'], None]

    wrong_csvs = [
        Path("03_IE_ner/check_study_type/animal_studies_case_report_publications.csv"),
        Path("03_IE_ner/check_study_type/animal_studies_review_publications.csv"),
        Path("03_IE_ner/check_study_type/animal_studies_clinical_trial_publications.csv"),
    ]
    EXCLUDE_PMIDS = load_wrong_pmids(wrong_csvs)

    OUTPUT = Path(BASE_DIR) / 'combined'
    combine_all(
        base_dir=BASE_DIR,
        subfolders=SUBFOLDERS,
        inner_folders=INNER_FOLDERS,
        exclude_pmids=EXCLUDE_PMIDS,
        output_path=OUTPUT
    )
