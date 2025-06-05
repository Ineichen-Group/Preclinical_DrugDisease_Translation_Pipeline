import re
from bs4 import BeautifulSoup
import os
import pandas as pd
import zipfile
from io import TextIOWrapper
import json
from section_detection_rules import is_start_of_materials_methods, is_end_of_materials_methods

def extract_from_standard_sections(soup, doc_id):
    """
    Extract paragraphs from sections with headings related to 'Materials and Methods'.
    
    This strategy uses:
        - <section> tags with a data-title attribute
        - <section> or <div> blocks with a matching top-level heading
        - <section> tags with class 'article-section__content'

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique document identifier.

    Returns:
        list[dict] | None: Extracted paragraph data or None if not found.
    """
    candidate_blocks = []

    # Strategy 1: Match <section data-title="...">
    section = soup.find("section", {"data-title": re.compile(r"materials\s*(and|&)?\s*methods", re.IGNORECASE)})
    if section:
        candidate_blocks.append(section)

    # Strategy 2: Match heading text in <section> or <div>
    for block in soup.find_all(["section", "div"]):
        heading = block.find(["h1", "h2", "h3"], recursive=False)
        if heading and is_start_of_materials_methods(heading.get_text(strip=True)):
            candidate_blocks.append(block)

    # Strategy 3: Match class-based section layout
    for block in soup.find_all("section", class_="article-section__content"):
        h2 = block.find("h2", class_="article-section__title")
        if h2 and is_start_of_materials_methods(h2.get_text(strip=True)):
            candidate_blocks.append(block)

    if not candidate_blocks:
        return None

    paragraphs = []
    for section in candidate_blocks:
        current_subtitle = "Materials and Methods"

        for tag in section.find_all(["h2", "h3", "h4", "p"]):
            text = tag.get_text(strip=True)

            if is_end_of_materials_methods(text):
                break

            if tag.name in {"h2", "h3", "h4"}:
                current_subtitle = text.rstrip(".:")
            elif tag.name == "p" and text:
                paragraphs.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": text
                })

    return paragraphs if paragraphs else None


def extract_from_fallback_divs(soup, doc_id):
    """
    Attempts to extract content from fallback <div> elements or anchored sections.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique document identifier.

    Returns:
        list[dict] | None: Extracted paragraph data or None if not found.
    """
    div_candidates = ["methods", "section-2"]
    target_div = None

    for div_id in div_candidates:
        target_div = soup.find("div", {"id": div_id})
        if target_div:
            break

    if not target_div:
        for anchor in soup.find_all("a", href=True):
            if is_start_of_materials_methods(anchor.get_text(strip=True)):
                ref_id = anchor["href"].lstrip("#")
                target_div = soup.find("div", {"id": ref_id})
                if target_div:
                    break

    if not target_div:
        return None

    paragraphs = []
    current_subtitle = "Methods"

    for tag in target_div.find_all(["p", "span"]):
        text = tag.get_text(strip=True)
        if is_end_of_materials_methods(text):
            break

        if tag.name == "span" and "level-4" in tag.get("class", []):
            current_subtitle = text
        elif tag.name == "p" and text:
            paragraphs.append({
                "doc_id": doc_id,
                "subtitle": current_subtitle,
                "paragraph": text
            })

    return paragraphs if paragraphs else None


def extract_from_ovid_format(soup, doc_id):
    """
    Extracts content from OVID-formatted HTML structure.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique document identifier.

    Returns:
        list[dict] | None: Extracted paragraph data or None if not found.
    """
    for h2 in soup.find_all("h2", class_="ejp-article-outline-heading"):
        if is_start_of_materials_methods(h2.get_text(strip=True)):
            content_blocks = []
            sibling = h2.find_next_sibling()
            while sibling and not (sibling.name == "h2" and "ejp-article-outline-heading" in sibling.get("class", [])):
                content_blocks.append(sibling)
                sibling = sibling.find_next_sibling()

            paragraphs = []
            current_subtitle = "Materials and Methods"
            for block in content_blocks:
                if block.name == "p":
                    text = block.get_text(strip=True)
                    if is_end_of_materials_methods(text):
                        break

                    strong_em = block.find("strong")
                    if strong_em and strong_em.find("em"):
                        current_subtitle = strong_em.get_text(strip=True).rstrip(":")

                    if text:
                        paragraphs.append({
                            "doc_id": doc_id,
                            "subtitle": current_subtitle,
                            "paragraph": text
                        })
            return paragraphs if paragraphs else None

    return None


def extract_by_heading_walk(soup, doc_id):
    """
    Fallback strategy: Walks forward from a heading that matches 'Materials and Methods',
    capturing sibling <h3>/<h4> subtitles and <p> paragraphs.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique document identifier.

    Returns:
        list[dict] | None: Paragraph rows with subtitles or None if nothing extracted.
    """
    # Find the starting heading (usually <h2>) that signals the start of the Methods section
    methods_heading = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        if is_start_of_materials_methods(tag.get_text(strip=True)):
            methods_heading = tag
            break

    if not methods_heading:
        return None

    rows = []
    current_subtitle = "Materials and Methods"
    current = methods_heading.find_next_sibling()

    while current:
        # Stop if we've reached another major section heading
        if current.name in {"h1", "h2"} and is_end_of_materials_methods(current.get_text(strip=True)):
            break

        if current.name in {"h3", "h4"}:
            current_subtitle = current.get_text(strip=True).rstrip(":.")

        elif current.name == "p":
            p_text = current.get_text(strip=True)
            if p_text and not is_end_of_materials_methods(p_text):
                rows.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": p_text
                })

        current = current.find_next_sibling()

    return rows if rows else None

def extract_from_nlm_structured_divs(soup, doc_id):
    """
    Extracts methods section from deeply nested div structures (e.g., NLM_sec).
    Designed for articles using structured div tags with class-based semantics.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Document ID.

    Returns:
        list[dict] | None: Extracted paragraphs with subtitles.
    """
    # Find the top-level <div> section with an <h2> matching Materials and Methods
    methods_container = None
    for section in soup.find_all("div", class_="NLM_sec NLM_sec_level_1"):
        heading = section.find("h2")
        if heading and is_start_of_materials_methods(heading.get_text(strip=True)):
            methods_container = section
            break

    if not methods_container:
        return None

    rows = []
    current_subtitle = "Materials and Methods"

    # Search inside nested NLM_sec_level_2 sections
    for subsection in methods_container.find_all("div", class_="NLM_sec NLM_sec_level_2", recursive=True):
        h3 = subsection.find(["h3", "h4"])
        if h3:
            current_subtitle = h3.get_text(strip=True).rstrip(":.")

        for para in subsection.find_all("div", class_="NLM_p", recursive=True):
            text = para.get_text(strip=True)
            if text and not is_end_of_materials_methods(text):
                rows.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": text
                })

    return rows if rows else None

def extract_from_semantic_sections_with_roles(soup, doc_id):
    """
    Extracts methods section from semantic <section> blocks using role-based content divs.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique identifier.

    Returns:
        list[dict] | None: Extracted paragraphs with subtitles or None.
    """
    methods_section = soup.find("section", {"data-type": re.compile(r"materials\s*methods", re.IGNORECASE)})
    if not methods_section:
        return None

    rows = []
    current_subtitle = "Materials and Methods"

    # Iterate over all nested sections inside the main methods section
    for subsec in methods_section.find_all("section", recursive=True):
        heading = subsec.find(["h3", "h4"])
        if heading:
            current_subtitle = heading.get_text(strip=True).rstrip(":.")

        for para_div in subsec.find_all("div", attrs={"role": "paragraph"}):
            text = para_div.get_text(strip=True)
            if text and not is_end_of_materials_methods(text):
                rows.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": text
                })

    return rows if rows else None

def extract_materials_methods_to_csv(soup, doc_id, output_csv_path, logs_dir=None):
    """
    Attempts multiple strategies to extract 'Materials and Methods' content from HTML.

    Parameters:
        soup (BeautifulSoup): Parsed HTML document.
        doc_id (str): Unique identifier of the document.
        output_csv_path (str): Path to output CSV file.
        logs_dir (str | None): Directory to write failure logs, if provided.

    Returns:
        tuple[bool, int]: (Success status, Number of unique subtitles extracted)
    """
    extraction_strategies = [
        extract_from_standard_sections,
        extract_from_semantic_sections_with_roles,
        extract_from_nlm_structured_divs,
        extract_from_fallback_divs,
        extract_from_ovid_format,
        extract_by_heading_walk
    ]

    paragraphs = None
    for strategy in extraction_strategies:
        paragraphs = strategy(soup, doc_id)
        if paragraphs:
            print(f"[INFO] Extracted using: {strategy.__name__}")
            break

    if not paragraphs:
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, "no_methods_docs_html.txt"), "a") as log_file:
                log_file.write(f"{doc_id}\n")
        return False, 0

    df = pd.DataFrame(paragraphs, columns=["doc_id", "subtitle", "paragraph"])
    df = df.groupby(["doc_id", "subtitle"], as_index=False).agg({
        "paragraph": lambda x: "\n\n".join(x)
    })

    df.to_csv(output_csv_path, mode='a', index=False, header=not os.path.exists(output_csv_path))
    return True, df["subtitle"].nunique()

def process_html_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Assuming there's only one HTML file inside the zip
        html_filename = [f for f in z.namelist() if f.endswith('.html')][0]
        with z.open(html_filename) as html_file:
            soup = BeautifulSoup(TextIOWrapper(html_file, 'utf-8'), 'html.parser')
            return soup

def process_each_html_in_df(metadata_retrieved_df_html, cadmus_outputs_dir, output_dir, logs_dir):
    success = 0
    total = 0
    subtitle_counts = []
    for idx, row in metadata_retrieved_df_html.iterrows():
        zip_path = row["html_parse_d"]["file_path"]  # or row["html_parse_d"].get("file_path") if it's a dict
        pmid = row["pmid"]
        zip_path = zip_path.replace("output", f"07_full_text_retrieval/cadmus/{cadmus_outputs_dir}")
        if not os.path.exists(zip_path):
            print(f"File not found: {zip_path}")
            continue
    
        try:
            soup = process_html_from_zip(zip_path)
            saved, unique_subtitles = extract_materials_methods_to_csv(soup, pmid, output_csv_path=f"{output_dir}/methods_subtitles_{pmid}.csv", logs_dir=logs_dir)
            success += 1 if saved else 0
            total += 1
            if saved:
                subtitle_counts.append(unique_subtitles)
        except Exception as e:
            print(f"Error processing {zip_path}: {e}")
            
    success_rate = (success / total) * 100 if total else 0
    avg_subtitles = sum(subtitle_counts) / len(subtitle_counts) if subtitle_counts else 0

    print(f"\nSummary Statistics {cadmus_outputs_dir}:")
    print(f"  Total PMIDs processed    : {total}")
    print(f"  Successful methods extractions   : {success}")
    print(f"  Success rate             : {success_rate:.2f}%")
    print(f"  Avg. unique subtitles    : {avg_subtitles:.2f}")
    # Save summary stats
    summary_path = os.path.join(logs_dir, "summary_stats_html_methods.txt")
    with open(summary_path, "a") as f:
        f.write(f"Summary Statistics {cadmus_outputs_dir}:\n")
        f.write(f"Total PMIDs processed  : {total}\n")
        f.write(f"Successful methods extractions : {success}\n")
        f.write(f"Success rate           : {success_rate:.2f}%\n")
        f.write(f"Avg. unique subtitles  : {avg_subtitles:.2f}")
        
def process_cadmus_output(cadmus_outputs_dir="output_UoZ"):
    with zipfile.ZipFile(f"07_full_text_retrieval/cadmus/{cadmus_outputs_dir}/retrieved_df/retrieved_df2.json.zip", "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                data = json.loads(data)
    f.close()
    z.close()

    metadata_retrieved_df = pd.read_json(data, orient='index')
    metadata_retrieved_df.pmid = metadata_retrieved_df.pmid.astype(str)
    metadata_retrieved_df_html = metadata_retrieved_df[metadata_retrieved_df['html']==1][['pmid','html_parse_d']]

    process_each_html_in_df(
        metadata_retrieved_df_html,
        cadmus_outputs_dir,
        output_dir="07_full_text_retrieval/materials_methods/from_html/MS_methods/",
        logs_dir="07_full_text_retrieval/materials_methods/logs/"
    )

def main():
    process_cadmus_output(cadmus_outputs_dir="output_UoZ")
    process_cadmus_output(cadmus_outputs_dir="output_UoE")

if __name__ == "__main__":
    main()