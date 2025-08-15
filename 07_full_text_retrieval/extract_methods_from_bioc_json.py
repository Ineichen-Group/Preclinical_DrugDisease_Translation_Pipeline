import json
from pathlib import Path
import pandas as pd
from typing import Tuple
from cadmus_extractors.xml_extractor import (
    extract_methods as xml_extract,
)
import argparse

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

        for p in passages:
            infons = p.get("infons", {})
            section_type = infons.get("section_type", "").upper()
            type_ = infons.get("type", "").lower()
            text = p.get("text", "").strip()

            if section_type == "METHODS":

                if type_ == "title_2":
                    # Treat level-2 titles as subtitles under METHODS
                    current_subtitle = text or "METHODS"
                elif type_ == "paragraph" and text:
                    rows.append({
                        "doc_id": pmid,
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
            no_methods_file = logs_dir / "no_methods_docs_pmc.txt"
            with no_methods_file.open("a") as f:
                f.write(f"{pmid}\n")
        return False, 0

def extract_methods_subtitles_to_json(
        json_path: Path,
        output_json: Path,
        logs_dir: Path,
        disease: str = "all_pmids"
    ) -> Tuple[bool, int]:
    """
    Parse a PMC JSON file and extract all 'Methods' section titles and paragraphs.

    Parameters:
        json_path (Path): Path to the PMC JSON file.
        output_json (Path): Path where the extracted JSON will be saved.
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

        for p in passages:
            infons = p.get("infons", {})
            section_type = infons.get("section_type", "").upper()
            type_ = infons.get("type", "").lower()
            text = p.get("text", "").strip()

            if section_type == "METHODS":  

                if type_ == "title_2":
                    # Treat level-2 titles as subtitles under METHODS
                    current_subtitle = text or "METHODS"
                elif type_ == "paragraph" and text:
                    rows.append({
                        "doc_id": pmid,
                        "subtitle": current_subtitle,
                        "paragraph": text
                    })
            else:
                continue

    # Write results to JSON if any rows collected
    if rows:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        unique_subtitles = len(set(row["subtitle"] for row in rows))
        print(f"Saved {len(rows)} entries to {output_json} ({unique_subtitles} unique subtitles)")
        return True, unique_subtitles
    else:
        print(f"No methods found in {json_path.name}")
        if logs_dir:
            logs_dir.mkdir(parents=True, exist_ok=True)
            no_methods_file = logs_dir / f"no_methods_docs_pmc_{disease}.txt"
            with no_methods_file.open("a") as f:
                f.write(f"{pmid}\n")
        return False, 0

def process_each_json_or_xml_in_dir(
    json_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    disease: str
) -> None:
    """
    Iterate over all JSON and XML files in a directory, extracting methods sections to JSON.

    Parameters:
        json_dir (Path): Directory containing PMC JSON or XML files (one per PMID).
        output_dir (Path): Directory where per-PMID outputs will be created.
        logs_dir (Path): Directory where summary and error logs go.
        disease (str): Label used when writing summary stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    success = 0
    subtitle_counts = []

    json_total = 0
    xml_total = 0

    for file in json_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".json":
            pmid = file.stem
            total += 1
            json_total += 1

            output_json = output_dir / f"methods_subtitles_{pmid}.json"

            saved, unique_subtitles = extract_methods_subtitles_to_json(
                json_path=file,
                output_json=output_json,
                logs_dir=logs_dir,
                disease=disease
            )

            if saved:
                success += 1
                subtitle_counts.append(unique_subtitles)

        elif file.is_file() and file.suffix.lower() == ".xml":
            print(f"Processing XML file: {file.name}")
            pmid = file.stem
            total += 1
            xml_total += 1

            saved, unique_subtitles = xml_extract(
                file_path=file,
                pmid=pmid,
                parse_info="",
                output_dir=output_dir,
                logs_dir=logs_dir
            )

            if saved:
                success += 1
                subtitle_counts.append(unique_subtitles)

    # Calculate summary statistics
    success_rate = (success / total) * 100 if total else 0.0
    avg_subtitles = (sum(subtitle_counts) / len(subtitle_counts)) if subtitle_counts else 0.0

    summary_text = (
        f"\nSummary Statistics ({disease}):\n"
        f"  Total files processed      : {total}\n"
        f"    JSON files               : {json_total}\n"
        f"    XML files                : {xml_total}\n"
        f"  Successful extractions     : {success}\n"
        f"  Success rate               : {success_rate:.2f}%\n"
        f"  Avg. unique subtitles      : {avg_subtitles:.2f}\n"
    )

    # (optional) still print if you like
    print(summary_text, end="")

    return summary_text


def main() -> None:
    """
    CLI entry point to process all *_fulltext folders inside a given input base directory.

    Example:
        --input_base 07_full_text_retrieval/pmc_fulltext

        Will process:
            - 07_full_text_retrieval/pmc_fulltext/parkinson_fulltext
            - 07_full_text_retrieval/pmc_fulltext/alzheimer_fulltext
            - etc.

        And write to:
            - 07_full_text_retrieval/materials_methods/bioc_json/parkinson_methods
            - 07_full_text_retrieval/materials_methods/bioc_json/alzheimer_methods
            ...
    """
    parser = argparse.ArgumentParser(
        description="Extract Materials & Methods sections from all *_fulltext folders."
    )

    parser.add_argument(
        "--input_base",
        type=Path,
        default=Path("07_full_text_retrieval/pmc_fulltext"),
        help="Base input directory containing *_fulltext subfolders."
    )

    parser.add_argument(
        "--output_base",
        type=Path,
        default=Path("07_full_text_retrieval/materials_methods/bioc_json"),
        help="Base output directory to save *_methods folders."
    )

    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("07_full_text_retrieval/materials_methods/logs/pmc"),
        help="Directory to store logs and summaries."
    )

    args = parser.parse_args()

    input_base = args.input_base
    print(f"Scanning for *_fulltext folders in: {input_base}")
    summaries = []

    for subfolder in sorted(input_base.glob("*_fulltext")):
        if not subfolder.is_dir():
            continue

        folder_domain = subfolder.name.replace("_fulltext", "")
        print(f"\nProcessing domain: {folder_domain}")

        input_path = subfolder
        output_path = args.output_base / f"{folder_domain}_methods"

        summary_text = process_each_json_or_xml_in_dir(
            json_dir=input_path,
            output_dir=output_path,
            logs_dir=args.logs_dir,
            disease=folder_domain  # used internally for log naming
        )
        summaries.append(summary_text)
        
    combined = args.logs_dir / f"summary_stats_pmc_methods_all.txt"
    combined.write_text("\n".join(summaries))
    
    print(f"\nWrote combined summary: {combined}")

if __name__ == "__main__":
    main()
