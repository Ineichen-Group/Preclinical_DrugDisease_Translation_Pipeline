import zipfile
import re
import xml.etree.ElementTree as ET
from io import TextIOWrapper
import os
import pandas as pd
from section_detection_rules import is_start_of_materials_methods, is_end_of_materials_methods
import json

def print_raw_xml(zip_path, max_chars=None):
    """
    Prints the raw XML content from the first .xml file in the given ZIP archive.

    Parameters:
        zip_path (str): Path to the ZIP file.
        max_chars (int | None): Optionally limit the number of characters printed.
    """
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as z:
        xml_files = [f for f in z.namelist() if f.endswith('.xml')]
        if not xml_files:
            print("[ERROR] No .xml file found in the ZIP.")
            return

        xml_filename = xml_files[0]
        with z.open(xml_filename) as xml_file:
            content = xml_file.read().decode('utf-8')
            if max_chars:
                print(content[:max_chars])
            else:
                print(content)

def extract_materials_methods_from_xml(zip_path, doc_id):
    """
    Extracts 'Materials and Methods' section from a Wiley-like XML file inside a ZIP archive.

    Detects XML namespace, finds the first section with a matching methods-like title, 
    and collects all paragraphs until a known stop section is reached.

    Parameters:
        zip_path (str): Path to ZIP file containing XML.
        doc_id (str): Document ID (e.g., PMID).

    Returns:
        list[dict] | None: List of extracted paragraphs with subtitles, or None if none found.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Assume only one XML file in the ZIP
        xml_filename = [f for f in z.namelist() if f.endswith('.xml')][0]
        with z.open(xml_filename) as xml_file:
            tree = ET.parse(TextIOWrapper(xml_file, 'utf-8'))
            root = tree.getroot()

            # Detect namespace from root tag (e.g., {http://www.wiley.com/namespaces/wiley})
            m = re.match(r'\{(.*)\}', root.tag)
            ns_uri = m.group(1) if m else ''
            has_ns = bool(ns_uri)
            ns = {'ns': ns_uri} if has_ns else {}

            # Prepare tag paths based on namespace usage
            section_path = './/ns:section' if has_ns else './/section'
            title_tag = 'ns:title' if has_ns else 'title'
            p_tag = 'ns:p' if has_ns else 'p'

            results = []
            in_methods = False
            current_subtitle = "Materials and Methods"

            # Loop through all <section> elements
            for section in root.findall(section_path, ns):
                # Get section title safely
                title_elem = section.find(title_tag, ns) if has_ns else section.find(title_tag)
                title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                # Detect entry into Materials and Methods section
                if not in_methods and is_start_of_materials_methods(title_text):
                    in_methods = True
                    current_subtitle = title_text or current_subtitle
                # Exit when hitting a stop section like Results, Discussion, etc.
                elif in_methods and is_end_of_materials_methods(title_text):
                    break
                # For subsections within methods, keep updating subtitle
                elif in_methods:
                    current_subtitle = title_text or current_subtitle

                # Extract paragraph text within methods section
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
            
def extract_from_jats_like_article(zip_path, doc_id):
    """
    Extracts 'Materials and Methods' sections from a JATS-style XML file inside a ZIP archive.

    Finds the first section with a methods-like title, then recursively collects text from it and nested subsections,
    stopping at known section titles like "Results" or "Discussion".

    Parameters:
        zip_path (str): Path to the ZIP file containing the XML.
        doc_id (str): Document identifier for tracking.

    Returns:
        list[dict] | None: List of extracted paragraphs with subtitles, or None if nothing found.
    """

    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get the first XML file from the ZIP archive
        xml_filename = [f for f in z.namelist() if f.endswith('.xml')][0]
        with z.open(xml_filename) as xml_file:
            tree = ET.parse(TextIOWrapper(xml_file, 'utf-8'))
            root = tree.getroot()

            # Detect default namespace, if any
            m = re.match(r'\{(.*)\}', root.tag)
            ns_uri = m.group(1) if m else ''
            ns = {'ns': ns_uri} if ns_uri else {}
            use_ns = bool(ns_uri)

            def tag(name):
                # Helper to generate namespaced or plain tag names
                return f'ns:{name}' if use_ns else name

            # Find the main <body> element
            body = root.find(f'.//{tag("body")}', ns)
            if body is None:
                return None

            results = []

            def parse_sec(sec, current_subtitle):
                """Recursively parse a <sec> section and collect paragraphs."""
                # Get and clean the title
                title_elem = sec.find(tag('title'), ns)
                title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                # Stop if a non-methods section title is reached
                if is_end_of_materials_methods(title_text):
                    return False

                # Use this section’s title as subtitle, or inherit from parent
                subtitle = title_text or current_subtitle

                # Extract paragraphs
                for p in sec.findall(tag('p'), ns):
                    text = "".join(p.itertext()).strip()
                    if text:
                        results.append({
                            "doc_id": doc_id,
                            "subtitle": subtitle,
                            "paragraph": text
                        })

                # Recurse into any child sections
                for child_sec in sec.findall(tag('sec'), ns):
                    if parse_sec(child_sec, subtitle) is False:
                        return False

                return True

            def find_first_methods_sec(node):
                """Search recursively for the first methods-like section."""
                for sec in node.findall(tag('sec'), ns):
                    title_elem = sec.find(tag('title'), ns)
                    title_text = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

                    # If this is a methods section, start parsing
                    if is_start_of_materials_methods(title_text):
                        return parse_sec(sec, title_text)

                    # Otherwise, keep searching deeper
                    if find_first_methods_sec(sec):
                        return True
                return False

            # Start looking from the body element
            if find_first_methods_sec(body):
                return results if results else None
            return None


def extract_materials_methods_from_xml_to_csv(zip_path, doc_id, output_csv_path, logs_dir=None):
    """
    Extracts 'Materials and Methods' section from XML in ZIP and saves to CSV.
    Falls back to JATS-style extraction if initial extraction fails.

    Parameters:
        zip_path (str): Path to ZIP file containing XML.
        doc_id (str): Unique document ID.
        output_csv_path (str): Where to save output CSV.
        logs_dir (str | None): Optional log directory for failures.

    Returns:
        tuple[bool, int]: (Success, number of unique subtitles)
    """
    paragraphs = extract_materials_methods_from_xml(zip_path, doc_id)

    # Fallback to JATS-style if initial extraction fails
    if not paragraphs:
        print(f"[INFO] Fallback to JATS extractor for {doc_id}")
        paragraphs = extract_from_jats_like_article(zip_path, doc_id)

    if not paragraphs:
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, "no_methods_docs_xml.txt"), "a") as log_file:
                log_file.write(f"{doc_id}\n")
        return False, 0

    df = pd.DataFrame(paragraphs, columns=["doc_id", "subtitle", "paragraph"])
    df = df.groupby(["doc_id", "subtitle"], as_index=False).agg({
        "paragraph": lambda x: "\n\n".join(x)
    })

    df.to_csv(output_csv_path, mode='a', index=False, header=not os.path.exists(output_csv_path))
    return True, df["subtitle"].nunique()

def process_each_xml_in_df(metadata_retrieved_df_xml, cadmus_outputs_dir, output_dir, logs_dir):
    success = 0
    total = 0
    subtitle_counts = []

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in metadata_retrieved_df_xml.iterrows():
        zip_path = row["xml_parse_d"]["file_path"]
        pmid = row["pmid"]
        zip_path = zip_path.replace("output", f"07_full_text_retrieval/cadmus/{cadmus_outputs_dir}")
        #print_raw_xml(zip_path, max_chars=50000)
        output_csv_path = f"{output_dir}/methods_subtitles_{pmid}.csv"

        if not os.path.exists(zip_path):
            print(f"[WARNING] File not found: {zip_path}")
            if logs_dir:
                os.makedirs(logs_dir, exist_ok=True)
                with open(os.path.join(logs_dir, "missing_files.txt"), "a") as log_file:
                    log_file.write(f"{pmid}\t{zip_path}\n")
            continue

        total += 1

        try:
            was_successful, num_subtitles = extract_materials_methods_from_xml_to_csv(
                zip_path=zip_path,
                doc_id=pmid,
                output_csv_path=output_csv_path,
                logs_dir=logs_dir
            )

            if was_successful:
                success += 1
                subtitle_counts.append(num_subtitles)
            else:
                print(f"[INFO] No methods found for {pmid}")
        except Exception as e:
            print(f"[ERROR] Failed to process {pmid}: {e}")
            if logs_dir:
                with open(os.path.join(logs_dir, "processing_errors.txt"), "a") as log_file:
                    log_file.write(f"{pmid}\t{zip_path}\t{str(e)}\n")

    success_rate = (success / total) * 100 if total else 0
    avg_subtitles = sum(subtitle_counts) / len(subtitle_counts) if subtitle_counts else 0

    print("\nSummary Statistics:")
    print(f"  Total PMIDs processed    : {total}")
    print(f"  Successful methods extractions   : {success}")
    print(f"  Success rate             : {success_rate:.2f}%")
    print(f"  Avg. unique subtitles    : {avg_subtitles:.2f}")

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
    metadata_retrieved_df_xml = metadata_retrieved_df[metadata_retrieved_df['xml']==1][['pmid','xml_parse_d']]
    
    # Step 1: Load the wrongly classified PubMed study type files
    wrong_files = [
        "03_IE_ner/check_study_type/animal_studies_case_report_publications.csv",
        "03_IE_ner/check_study_type/animal_studies_review_publications.csv",
        "03_IE_ner/check_study_type/animal_studies_clinical_trial_publications.csv"
    ]

    # Combine all PMIDs from the incorrect study types into a single set
    wrong_pmids = set()
    for file in wrong_files:
        df = pd.read_csv(file)
        wrong_pmids.update(df['PMID'].astype(str))

    filtered_df = metadata_retrieved_df_xml[~metadata_retrieved_df_xml['pmid'].astype(str).isin(wrong_pmids)]
    print(f"Filtered out {len(metadata_retrieved_df_xml) - len(filtered_df)} wrong PMIDs, remaining {len(filtered_df)} studies.")

    process_each_xml_in_df(
            metadata_retrieved_df_xml=filtered_df,
            cadmus_outputs_dir=cadmus_outputs_dir,
            output_dir="07_full_text_retrieval/materials_methods/from_xml/MS_methods/",
            logs_dir="07_full_text_retrieval/materials_methods/logs/"
        )

def main():
    process_cadmus_output(cadmus_outputs_dir="output_UoZ")
    process_cadmus_output(cadmus_outputs_dir="output_UoE")

if __name__ == "__main__":
    main()