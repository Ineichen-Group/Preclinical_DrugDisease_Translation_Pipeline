from pathlib import Path
import pandas as pd
from typing import List, Union, Set, Iterable, Dict, Any
from cadmus_extractors.utils import load_wrong_pmids
import csv
import re
import argparse
import json, os, logging


# Configure a module-level logger once
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# If no handlers are set, add a basic one (so it works stand-alone too)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

def iter_json_file(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Stream records from either a JSON array file or JSON Lines (JSONL).
    Uses ijson for arrays if available; otherwise falls back to loading the file.
    """
    try:
        import ijson  # pip install ijson
    except Exception:
        ijson = None

    with path.open("rb") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith(b"["):  # JSON array
            if ijson is None:
                # Fallback only OK if files are not huge
                for rec in json.load(f):
                    yield rec
            else:
                for rec in ijson.items(f, "item"):
                    yield rec
        else:  # JSONL
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def aggregate_one_file(json_path: Path, include_subtitles: bool = True) -> Dict[str, str]:
    """
    Read ONE JSON file, group by doc_id/pmid, and return {pmid: concatenated_text}.
    Keeps only in-memory buckets for this single file.
    """
    buckets: Dict[str, list] = {}
    n_records = 0

    for rec in iter_json_file(json_path):
        n_records += 1
        pmid = str(rec.get("PMID") or rec.get("pmid") or rec.get("doc_id") or "").strip()
        if not pmid:
            continue
        para = (rec.get("paragraph") or rec.get("Text") or rec.get("text") or "").strip()
        if not para:
            continue
        if include_subtitles:
            sub = (rec.get("subtitle") or "").strip()
            chunk = f"{sub}\n{para}" if sub else para
        else:
            chunk = para
        buckets.setdefault(pmid, []).append(chunk)

    doc_texts = {pmid: "\n\n".join(chunks) for pmid, chunks in buckets.items()}
    logger.info(f"{json_path.name}: read {n_records:,} rows; aggregated {len(doc_texts):,} docs")
    return doc_texts

# --------------- main combiner -----------------
def combine_all(
    base_dir: str,
    subfolders: List[str],
    inner_folders: List[Union[None, str, List[str]]],
    exclude_pmids: Set[str],
    output_file: str,
    flush_every: int = 5000,
    include_subtitles: bool = True,
):
    """
    File-by-file processing:
      - For each JSON file, aggregate paragraphs per doc_id.
      - Append aggregated docs to output JSONL immediately.
      - Continue to the next file (low memory).
    Output schema per line: {"PMID": <str>, "Text": <str>, "source_label": <str>}
    """
    base = Path(base_dir)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not base.exists():
        logger.warning(f"Skipping base_dir '{base_dir}' because it doesn't exist.")
        return

    seen_pmids: Set[str] = set()
    total_written = 0
    per_folder_stats: Dict[str, Dict[str, int]] = {}

    logger.info(f"Writing to: {out}")
    # Open once in "w" (fresh file), then append per record
    with out.open("w", encoding="utf-8") as w:
        for sub, inner in zip(subfolders, inner_folders):
            base_folder = base / sub
            logger.info(f"Processing subfolder: {sub}")

            if inner is None:
                folders_to_process = [base_folder]
            elif isinstance(inner, list):
                folders_to_process = [base_folder / i for i in inner]
            else:
                folders_to_process = [base_folder / inner]

            found_docs_for_sub = 0
            written_for_sub = 0

            for folder in folders_to_process:
                logger.info(f"  Scanning folder: {folder}")
                if not folder.is_dir():
                    logger.warning(f"Folder missing: {folder}")
                    continue

                json_files = sorted(folder.glob("*.json"))
                if not json_files:
                    logger.warning(f"No JSON files found in {folder}")
                    continue

                for jf in json_files:
                    try:
                        doc_texts = aggregate_one_file(jf, include_subtitles=include_subtitles)
                    except Exception as e:
                        logger.warning(f"Skipping {jf} due to error: {e}")
                        continue

                    found_docs_for_sub += len(doc_texts)

                    # Write each aggregated doc from this file, deduping globally
                    for pmid, text in doc_texts.items():
                        if pmid in exclude_pmids or pmid in seen_pmids:
                            continue
                        rec = {
                            "PMID": pmid,
                            "Text": normalize_text(text),
                            "source_label": sub,
                        }
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        seen_pmids.add(pmid)
                        written_for_sub += 1
                        total_written += 1

                        if total_written % flush_every == 0:
                            w.flush()
                            os.fsync(w.fileno())
                            logger.info(f"Wrote {total_written:,} total records so far...")

            per_folder_stats[sub] = {
                "found_docs": found_docs_for_sub,
                "written_unique": written_for_sub,
            }

        # final flush
        w.flush()
        os.fsync(w.fileno())

    # Summary
    logger.info("--- Per-folder (aggregated docs → written unique) ---")
    for sub, stats in per_folder_stats.items():
        logger.info(f"{sub:12s}: {stats['found_docs']:,} → {stats['written_unique']:,}")
    logger.info(f"Total written: {total_written:,}")
    logger.info(f"Saved combined JSONL to {out}")
    
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