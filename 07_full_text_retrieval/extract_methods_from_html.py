import re
from bs4 import BeautifulSoup
import os
import pandas as pd
import zipfile
from io import TextIOWrapper
import json

def find_methods_section(soup, doc_id):
    MATERIALS_METHODS_TITLES = [
        r"materials\s*(and|&)?\s*methods",
        r"methodology",
        r"experimental\s+(procedures|section)",
        r"\bmethods\b",
    ]

    section = None

    # 1. Try <section data-title="Materials and methods">
    section = soup.find("section", {"data-title": re.compile(r"materials\s*(and|&)?\s*methods", re.IGNORECASE)})

    # 2. Try <section> or <div> with an appropriate heading inside
    if not section:
        for block in soup.find_all(["section", "div"], recursive=True):
            heading = block.find(["h1", "h2", "h3"], recursive=False)
            if heading:
                heading_text = heading.get_text(strip=True).lower()
                if any(re.search(p, heading_text) for p in MATERIALS_METHODS_TITLES):
                    section = block
                    break

    # 3. Fallback: <section class="article-section__content"><h2 class="article-section__title">
    if not section:
        for sec in soup.find_all("section", class_="article-section__content"):
            h2 = sec.find("h2", class_="article-section__title")
            if h2:
                h2_text = h2.get_text(strip=True).lower()
                if any(re.search(p, h2_text) for p in MATERIALS_METHODS_TITLES):
                    section = sec
                    break

    # If we couldn't find the section, return None
    if not section:
        return None

    # Process the section content
    rows = []
    current_subtitle = "Materials and methods"

    for tag in section.find_all(["h2", "h3", "h4", "p"], recursive=True):
        tag_class = tag.get("class", [])
        tag_text = tag.get_text(strip=True)

        if tag.name in {"h2", "h3", "h4"} and (
            "c-article__sub-heading" in tag_class or
            "article-section__sub-title" in tag_class or
            not tag_class  # fallback to semantic level if no class
        ):
            current_subtitle = tag_text.rstrip(".:")
        elif tag.name == "p" and tag_text:
            rows.append({
                "doc_id": doc_id,
                "subtitle": current_subtitle,
                "paragraph": tag_text
            })

    return rows if rows else None


def fallback_extract_methods(soup, doc_id):
    fallback_section = None

    for id_candidate in ["methods", "section-2"]:
        fallback_section = soup.find("div", {"id": id_candidate})
        if fallback_section:
            break

    if not fallback_section:
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            if re.search(r"\bmethods\b", text):
                section_id = a["href"].lstrip("#")
                fallback_section = soup.find("div", {"id": section_id})
                if fallback_section:
                    break

    if not fallback_section:
        return None

    rows = []
    current_subtitle = "Methods"

    for tag in fallback_section.find_all(["p", "span"], recursive=True):
        if tag.name == "span" and "level-4" in tag.get("class", []):
            current_subtitle = tag.get_text(strip=True)
        elif tag.name == "p":
            text = tag.get_text(strip=True)
            if text:
                rows.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": text
                })
    return rows if rows else None

def extract_methods_ovid_style(soup, doc_id):
    MATERIALS_METHODS_TITLES = [
        r"materials\s*(and|&)?\s*methods",
        r"methodology",
        r"experimental\s+(procedures|section)",
        r"\bmethods\b",
    ]

    methods_h2 = None
    for h2 in soup.find_all("h2", class_="ejp-article-outline-heading"):
        h2_text = h2.get_text(strip=True).lower()
        if any(re.search(p, h2_text) for p in MATERIALS_METHODS_TITLES):
            methods_h2 = h2
            break

    if not methods_h2:
        return None

    # Gather elements until the next h2
    methods_content = []
    current = methods_h2.find_next_sibling()
    while current and (current.name != "h2" or "ejp-article-outline-heading" not in current.get("class", [])):
        methods_content.append(current)
        current = current.find_next_sibling()

    rows = []
    current_subtitle = "Materials and methods"

    for el in methods_content:
        if el.name == "p":
            strong_em = el.find("strong")
            if strong_em and strong_em.find("em"):
                current_subtitle = strong_em.get_text(strip=True).rstrip(":")
            text = el.get_text(strip=True)
            if text:
                rows.append({
                    "doc_id": doc_id,
                    "subtitle": current_subtitle,
                    "paragraph": text
                })

    return rows if rows else None

def extract_methods_subtitles_from_soup(soup, doc_id, output_csv, logs_dir=None):
    extractors = [
        find_methods_section,
        fallback_extract_methods,
        extract_methods_ovid_style
    ]

    rows = None
    for extractor in extractors:
        rows = extractor(soup, doc_id)
        if rows:
            break

    if not rows:
        if logs_dir:
            with open(os.path.join(logs_dir, "no_methods_docs_html.txt"), "a") as f:
                f.write(f"{doc_id}\n")
        return False, 0
        
    df = pd.DataFrame(rows, columns=["doc_id", "subtitle", "paragraph"])
    df = df.groupby(["doc_id", "subtitle"], as_index=False).agg({
        "paragraph": lambda paras: "\n\n".join(paras)
    })
    df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))
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
            saved, unique_subtitles = extract_methods_subtitles_from_soup(soup, pmid, output_csv=f"{output_dir}/methods_subtitles_{pmid}.csv", logs_dir=logs_dir)
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