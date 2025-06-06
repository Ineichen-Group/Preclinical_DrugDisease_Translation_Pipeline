# cadmus_processor/formats/xml_extractor.py

import os
import re
import zipfile
import logging
import xml.etree.ElementTree as ET
from io import TextIOWrapper
from pathlib import Path

import pandas as pd

from section_detection_rules import (
    is_start_of_materials_methods,
    is_end_of_materials_methods,
)
from cadmus_extractors.utils import setup_logger, ensure_dir


def can_handle(pmid: str, cadmus_base_dir: str, metadata_row: pd.Series) -> bool:
    """
    Return True if this row indicates an XML zip is available on disk.
    We expect metadata_row['xml'] == 1 and that metadata_row['xml_parse_d']['file_path'] exists.
    """
    if metadata_row.get("xml", 0) != 1:
        return False

    parse_info = metadata_row.get("xml_parse_d", {})
    zip_path = parse_info.get("file_path", "")
    zip_path = zip_path.replace("output", str(cadmus_base_dir))
    return bool(zip_path and os.path.exists(zip_path))


def extract_methods(
    pmid: str,
    cadmus_base_dir: Path,
    parse_info: dict,
    output_dir: Path,
    logs_dir: Path,
    logger: logging.Logger = None
) -> (bool, int): # type: ignore
    """
    Attempt to extract “Materials & Methods” paragraphs from an XML-in-ZIP.

    Parameters:
        pmid (str): Document identifier.
        parse_info (dict): Should contain {"file_path": "<path to zip>"}.
        output_dir (Path): Directory where we will write methods_subtitles_{pmid}.csv.
        logs_dir (Path): Directory where XML-specific logs (e.g. no_methods_docs_xml.txt) go.
        logger (logging.Logger): If None, create a local one.

    Returns:
        (was_successful: bool, num_unique_subtitles: int)
    """
    if logger is None:
        logger = setup_logger(__name__)

    zip_path = parse_info.get("file_path", "")
    zip_path = zip_path.replace("output", str(cadmus_base_dir))
    if not zip_path or not os.path.exists(zip_path):
        logger.warning(f"[XML][can_handle error] File not found for PMID {pmid}: {zip_path}")
        return False, 0

    ensure_dir(output_dir)
    ensure_dir(logs_dir)

    # Set up a per-module log file
    xml_log = logs_dir / "xml_processing.log"
    setup_logger(str(xml_log))

    # First attempt “Wiley-like” extraction
    try:
        paragraphs = _extract_materials_methods_from_xml(zip_path, pmid)
    except Exception as e:
        logger.error(f"[XML][ERROR] Wiley-like extraction failed for PMID {pmid}: {e}")
        paragraphs = None

    # If Wiley-like returns no paragraphs, attempt JATS-like
    if not paragraphs:
        logger.info(f"[XML] Falling back to JATS-like extractor for PMID {pmid}")
        try:
            paragraphs = _extract_from_jats_like_article(zip_path, pmid)
        except Exception as e:
            logger.error(f"[XML][ERROR] JATS-like extraction failed for PMID {pmid}: {e}")
            paragraphs = None

    if not paragraphs:
        # Nothing found → log and return failure
        _append_to_log(logs_dir / "no_methods_docs_xml.txt", pmid)
        return False, 0

    # Build DataFrame, group by (doc_id, subtitle)
    df = pd.DataFrame(paragraphs, columns=["doc_id", "subtitle", "paragraph"])
    df = (
        df
        .groupby(["doc_id", "subtitle"], as_index=False)
        .agg({"paragraph": lambda grp: "\n\n".join(grp)})
    )

    output_csv = output_dir / f"methods_subtitles_{pmid}.csv"
    header_flag = not output_csv.exists()
    df.to_csv(str(output_csv), mode="a", index=False, header=header_flag)

    num_unique = df["subtitle"].nunique()
    return True, num_unique


def _append_to_log(log_path: Path, text: str) -> None:
    """
    Append a single line to the given log file.
    """
    ensure_dir(log_path.parent)
    with open(str(log_path), "a") as f:
        f.write(f"{text}\n")


def _extract_materials_methods_from_xml(zip_path: str, doc_id: str):
    """
    Wiley-like XML extraction:
    - Detect namespace (if present).
    - Traverse <section> elements in order.
    - When a <section>'s <title> matches start‐of‐methods, set in_methods = True.
    - Collect all <p> under that section/subsections until an end-of-methods title is encountered.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        xml_files = [f for f in z.namelist() if f.lower().endswith(".xml")]
        if not xml_files:
            return None

        xml_filename = xml_files[0]
        with z.open(xml_filename) as xml_file:
            tree = ET.parse(TextIOWrapper(xml_file, "utf-8"))
            root = tree.getroot()

            # Detect namespace
            m = re.match(r"\{(.*)\}", root.tag)
            ns_uri = m.group(1) if m else ""
            has_ns = bool(ns_uri)
            ns = {"ns": ns_uri} if has_ns else {}

            section_path = ".//ns:section" if has_ns else ".//section"
            title_tag = "ns:title" if has_ns else "title"
            p_tag = "ns:p" if has_ns else "p"

            results = []
            in_methods = False
            current_subtitle = "Materials and Methods"

            for section in root.findall(section_path, ns):
                # Get section title
                title_elem = section.find(title_tag, ns) if has_ns else section.find(title_tag)
                title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                if not in_methods and is_start_of_materials_methods(title_text):
                    in_methods = True
                    current_subtitle = title_text or current_subtitle
                elif in_methods and is_end_of_materials_methods(title_text):
                    break
                elif in_methods:
                    current_subtitle = title_text or current_subtitle

                if in_methods:
                    for para in section.findall(p_tag, ns):
                        text = "".join(para.itertext()).strip()
                        if text:
                            results.append({
                                "doc_id": doc_id,
                                "subtitle": current_subtitle,
                                "paragraph": text
                            })

            return results if results else None


def _extract_from_jats_like_article(zip_path: str, doc_id: str):
    """
    JATS-like XML extraction:
    - Detect namespace.
    - Find <body> … then recursively search for the first <sec> whose <title> matches start‐of‐methods.
    - parse_sec(sec, current_subtitle) collects all <p> until an end-of-methods is found, then stops.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        xml_files = [f for f in z.namelist() if f.lower().endswith(".xml")]
        if not xml_files:
            return None

        xml_filename = xml_files[0]
        with z.open(xml_filename) as xml_file:
            tree = ET.parse(TextIOWrapper(xml_file, "utf-8"))
            root = tree.getroot()

            # Detect namespace
            m = re.match(r"\{(.*)\}", root.tag)
            ns_uri = m.group(1) if m else ""
            ns = {"ns": ns_uri} if ns_uri else {}
            use_ns = bool(ns_uri)

            def tag(name: str) -> str:
                return f"ns:{name}" if use_ns else name

            # Locate <body>
            body = root.find(f".//{tag('body')}", ns)
            if body is None:
                return None

            results = []

            def parse_sec(sec, current_subtitle):
                # Get section title
                title_elem = sec.find(tag("title"), ns)
                title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                # If this title is an end-of-methods marker, stop recursion
                if is_end_of_materials_methods(title_text):
                    return False

                subtitle = title_text or current_subtitle

                # Collect paragraphs
                for p in sec.findall(tag("p"), ns):
                    text = "".join(p.itertext()).strip()
                    if text:
                        results.append({
                            "doc_id": doc_id,
                            "subtitle": subtitle,
                            "paragraph": text
                        })

                # Recurse into nested <sec>
                for child_sec in sec.findall(tag("sec"), ns):
                    if parse_sec(child_sec, subtitle) is False:
                        return False

                return True

            def find_first_methods_sec(node):
                for sec in node.findall(tag("sec"), ns):
                    title_elem = sec.find(tag("title"), ns)
                    title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                    if is_start_of_materials_methods(title_text):
                        return parse_sec(sec, title_text)

                    if find_first_methods_sec(sec):
                        return True

                return False

            found = find_first_methods_sec(body)
            return results if found and results else None
