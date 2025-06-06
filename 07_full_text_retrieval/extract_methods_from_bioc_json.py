import os
import json
from pathlib import Path
import pandas as pd


def extract_methods_subtitles_to_csv(
    json_path: Path,
    output_csv: Path,
    logs_dir: Path
) -> (bool, int): # type: ignore
    """
    Parse a PMC JSON file and extract all 'Methods' section titles and paragraphs.

    Parameters:
        json_path (Path): Path to the PMC JSON file.
        output_csv (Path): Path where the extracted CSV will be saved.
        logs_dir (Path): Directory to append PMIDs for which no methods were found.

    Returns:
        Tuple[bool, int]:
            - True and number of unique subtitles if extraction succeeded.
            - False and 0 if no methods were found.
    """
    # Load JSON
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    pmid = None

    # The JSON structure is assumed to have a top-level list with 'documents'
    for doc in data[0].get("documents", []):
        passages = doc.get("passages", [])

        # Attempt to find PMID in passage infons
        for p in passages:
            pmid_candidate = p.get("infons", {}).get("article-id_pmid")
            if pmid_candidate:
                pmid = pmid_candidate
                break

        # If PMID wasn't found in infons, fall back to filename
        if not pmid:
            pmid = json_path.stem

        current_subtitle = "METHODS"
        in_methods = False

        for p in passages:
            infons = p.get("infons", {})
            section_type = infons.get("section_type", "").upper()
            type_ = infons.get("type", "").lower()
            text = p.get("text", "").strip()

            if section_type == "METHODS":
                in_methods = True

                if type_ == "title_2":
                    # Treat level-2 titles as subtitles under METHODS
                    current_subtitle = text or "METHODS"
                elif type_ == "paragraph" and text:
                    rows.append({
                        "pmid": pmid,
                        "subtitle": current_subtitle,
                        "paragraph": text
                    })
            else:
                # Skip passages not in METHODS section
                continue

    # Write results to CSV if any rows collected
    if rows:
        df = pd.DataFrame(rows, columns=["pmid", "subtitle", "paragraph"])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        unique_subtitles = df["subtitle"].nunique()
        print(f"Saved {len(rows)} rows to {output_csv} ({unique_subtitles} unique subtitles)")
        return True, unique_subtitles
    else:
        print(f"No methods found in {json_path.name}")
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            no_methods_file = logs_dir / "no_methods_pmids.txt"
            with no_methods_file.open("a") as f:
                f.write(f"{pmid}\n")
        return False, 0


def process_each_json_in_dir(
    json_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    disease: str
) -> None:
    """
    Iterate over all JSON files in a directory, extracting methods sections to CSV.

    Parameters:
        json_dir (Path): Directory containing PMC JSON files (one per PMID).
        output_dir (Path): Directory where per-PMID CSVs will be created.
        logs_dir (Path): Directory where summary and error logs go.
        disease (str): Label used when writing summary stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    total = 0
    subtitle_counts = []

    for json_file in json_dir.iterdir():
        if not json_file.is_file() or json_file.suffix.lower() != ".json":
            continue

        pmid = json_file.stem
        total += 1

        output_csv = output_dir / f"methods_subtitles_{pmid}.csv"
        saved, unique_subtitles = extract_methods_subtitles_to_csv(
            json_path=json_file,
            output_csv=output_csv,
            logs_dir=logs_dir
        )

        if saved:
            success += 1
            subtitle_counts.append(unique_subtitles)

    # Calculate summary statistics
    success_rate = (success / total) * 100 if total else 0.0
    avg_subtitles = (sum(subtitle_counts) / len(subtitle_counts)) if subtitle_counts else 0.0

    print(f"\nSummary Statistics ({disease}):")
    print(f"  Total JSON files processed : {total}")
    print(f"  Successful extractions     : {success}")
    print(f"  Success rate               : {success_rate:.2f}%")
    print(f"  Avg. unique subtitles      : {avg_subtitles:.2f}")

    # Write summary to log
    summary_path = logs_dir / f"summary_stats_pmc_methods_{disease}.txt"
    with summary_path.open("w") as f:
        f.write("Summary Statistics:\n")
        f.write(f"Total JSON files processed : {total}\n")
        f.write(f"Successful extractions     : {success}\n")
        f.write(f"Success rate               : {success_rate:.2f}%\n")
        f.write(f"Avg. unique subtitles      : {avg_subtitles:.2f}\n")


def main() -> None:
    """
    Example entry point to process PMC JSON files for a given disease.
    Change 'disease' to match the folder names for your JSON input.
    """
    disease = "epilepsy"  # or "parkinson"
    print(f"Processing '{disease}' methods extraction from PMC JSON files...")

    base_input = Path("07_full_text_retrieval/pmc_fulltext") / f"{disease}_fulltext"
    base_output = Path("07_full_text_retrieval/materials_methods/bioc_json") / f"{disease}_methods"
    logs_dir = Path("07_full_text_retrieval/materials_methods/logs")

    process_each_json_in_dir(
        json_dir=base_input,
        output_dir=base_output,
        logs_dir=logs_dir,
        disease=disease
    )

if __name__ == "__main__":
    main()
