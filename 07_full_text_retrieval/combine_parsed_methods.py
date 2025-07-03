from pathlib import Path
import pandas as pd
from typing import List, Optional, Set
from cadmus_extractors.utils import load_wrong_pmids

def process_standard_folder(
    folder: Path,
    exclude_pmids: Set[str],
    source_label: str,
    pattern: str = '*.csv'
) -> pd.DataFrame:
    """Return one big DataFrame of grouped articles for this folder."""
    frames = []
    for csv_path in folder.glob(pattern):
        df = pd.read_csv(csv_path)
        if 'pmid' not in df.columns and 'doc_id' in df.columns:
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
              .rename(columns={'pmid':'PMID'})
        )
        merged['Source'] = source_label
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['PMID','Text','Source'])

def process_pdf_folder(
    folder: Path,
    source_label: str,
    pattern: str = '*_full_text.csv'
) -> pd.DataFrame:
    """Return one big DataFrame of all *_full_text.csv for this folder."""
    frames = []
    for csv_path in folder.glob(pattern):
        df = pd.read_csv(csv_path)
        if 'doc_id' not in df.columns or 'Text' not in df.columns:
            print(f"Skipping {csv_path}: missing doc_id or Text")
            continue
        df = df.rename(columns={'doc_id':'PMID'})[['PMID','Text']]
        df['Source'] = source_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['PMID','Text','Source'])

def combine_all(
    base_dir: str,
    subfolders: List[str],
    inner_folders: List[Optional[str]],
    pdf_subfolder: str,
    exclude_pmids: Set[str],
    output_path: str
):
    base = Path(base_dir)
    all_frames = []
    folder_counts = {}  # count loaded PMIDs per subfolder

    for sub, inner in zip(subfolders, inner_folders):
        folder = base / sub
        if inner:
            folder = folder / inner
        if not folder.is_dir():
            print(f"Warning: {folder} missing")
            folder_counts[sub] = 0
            continue

        if sub == pdf_subfolder:
            df = process_pdf_folder(folder, source_label=sub)
        else:
            df = process_standard_folder(folder, exclude_pmids, source_label=sub)

        # tally how many unique PMIDs we got from this sub
        count = df['PMID'].nunique()
        folder_counts[sub] = count

        all_frames.append(df)

    # combine and dedupe overall
    combined = pd.concat(all_frames, ignore_index=True)
    print("\n--- Per-folder document counts ---")
    for sub, cnt in folder_counts.items():
        print(f"{sub:12s}: {cnt}")

    print(f"\nTotal before dedup: {combined['PMID'].nunique()} unique articles")
    combined = combined.drop_duplicates('PMID')
    print(f"Total after dedup : {combined['PMID'].nunique()} unique articles")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nSaved combined CSV to {out}, shape: {combined.shape}")

# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE_DIR      = '07_full_text_retrieval/materials_methods'
    SUBFOLDERS    = ['html', 'xml', 'plain', 'bioc_json', 'pdf']
    INNER_FOLDERS = [None,    None,    None,    'MS_methods', None]
    PDF_SUBFOLDER = 'pdf'

    wrong_csvs = [
        Path("03_IE_ner/check_study_type/animal_studies_case_report_publications.csv"),
        Path("03_IE_ner/check_study_type/animal_studies_review_publications.csv"),
        Path("03_IE_ner/check_study_type/animal_studies_clinical_trial_publications.csv"),
    ]
    EXCLUDE_PMIDS = load_wrong_pmids(wrong_csvs)

    OUTPUT = Path(BASE_DIR) / 'combined' / 'combined_methods.csv'
    combine_all(
        base_dir=BASE_DIR,
        subfolders=SUBFOLDERS,
        inner_folders=INNER_FOLDERS,
        pdf_subfolder=PDF_SUBFOLDER,
        exclude_pmids=EXCLUDE_PMIDS,
        output_path=str(OUTPUT)
    )
