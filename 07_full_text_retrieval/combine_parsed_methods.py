from pathlib import Path
import pandas as pd
from typing import List, Optional, Set, Union
from cadmus_extractors.utils import load_wrong_pmids
import csv
import re
import argparse
import json

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
    output_file: str
):
    base = Path(base_dir)
    all_frames = []
    folder_counts = {}
    
    if not base.exists():
        print(f"Skipping base_dir '{base_dir}' because it doesn't exist.")
        return
    
    for sub, inner in zip(subfolders, inner_folders):
        print(f"\nProcessing subfolder: {sub}")
        print(f"inner = {inner} ({type(inner)})")
    
        base_folder = base / sub

        # Handle single or multiple inner folders
        if inner is None:
            folders_to_process = [base_folder]
        elif isinstance(inner, list):
            folders_to_process = [base_folder / i for i in inner]
        else:  # string (not expected with the new JSON interface, but safe)
            folders_to_process = [base_folder / inner]
            
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

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    combined["Text"] = combined["Text"].map(normalize_text)

    combined.to_json(out, orient="records", lines=True, force_ascii=False)

    print(f"\nSaved combined CSV to {out}, shape: {combined.shape}")

if __name__ == '__main__':
    import argparse, json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Combine JSON article paragraphs into a single JSONL file.'
    )
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Root directory containing subfolders (base of inputs).')

    # NEW: explicit JSON for the two parallel lists
    parser.add_argument('--subfolders-json', type=str, required=True,
                        help='JSON array of subfolder names, e.g. ["cadmus","bioc_json"].')
    parser.add_argument('--inner-folders-json', type=str, required=True,
                        help='JSON array, same length as subfolders; each item is null or a list of inner folder names.')

    parser.add_argument('--exclude-csvs', type=str, nargs='*', default=[],
                        help='Paths to CSVs listing PMIDs to exclude')

    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the combined output JSONL file')

    args = parser.parse_args()

    # Parse JSON inputs
    try:
        subfolders = json.loads(args.subfolders_json)
        inner = json.loads(args.inner_folders_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON arguments: {e}")

    if not isinstance(subfolders, list) or not isinstance(inner, list):
        raise SystemExit("--subfolders-json and --inner-folders-json must be JSON arrays.")

    if len(subfolders) != len(inner):
        raise SystemExit(
            f"Length mismatch: subfolders ({len(subfolders)}) vs inner_folders ({len(inner)})."
        )

    # Validate inner structure (None or list[str])
    norm_inner = []
    for i, spec in enumerate(inner):
        if spec is None:
            norm_inner.append(None)
        elif isinstance(spec, list) and all(isinstance(x, str) for x in spec):
            norm_inner.append(spec)
        else:
            raise SystemExit(
                f"--inner-folders-json item {i} must be null or list[str]; got: {spec!r}"
            )

    # Load exclusion PMIDs
    wrong_csv_paths = [Path(p) for p in args.exclude_csvs]
    exclude_pmids = load_wrong_pmids(wrong_csv_paths)

    # Run combine
    combine_all(
        base_dir=args.base_dir,
        subfolders=subfolders,
        inner_folders=norm_inner,
        exclude_pmids=exclude_pmids,
        output_file=args.output_file
    )