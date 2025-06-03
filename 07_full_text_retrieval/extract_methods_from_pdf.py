from papermage.recipes import CoreRecipe
import re
import logging
import os 
import time
import pandas as pd
from collections import defaultdict
import zipfile
import json

logger = logging.getLogger("materials_methods_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevents duplicate log prints

def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def group_paragraphs_by_section(doc):
    """
    Groups paragraphs under their respective section headings based on character spans.

    Args:
        doc: A PaperMage Document object.

    Returns:
        A dictionary where each key is a section title (string) and each value is a list of
        paragraph texts (strings) belonging to that section.
        
    # Example usage:
     doc = recipe.run("path/to/your.pdf")
     section_dict = group_paragraphs_by_section(doc)
     print(section_dict)
    """
    # Build sorted list of (heading_start_offset, SectionEntity) pairs
    section_boundaries = sorted(
        [(sec.spans[0].start, sec) for sec in doc.sections],
        key=lambda x: x[0]
    )

    # Determine an "end-of-document" boundary for the last section
    max_para_end = max(para.spans[0].end for para in doc.paragraphs)
    section_boundaries.append((max_para_end + 1, None))

    # Initialize the result dict
    result = {}

    # Loop through each real section and collect paragraphs between this_heading and next_heading
    for idx in range(len(section_boundaries) - 1):
        this_start, sec_entity = section_boundaries[idx]
        next_start, _ = section_boundaries[idx + 1]

        # Collect all paragraphs where paragraph.spans[0].start is in [this_start, next_start)
        paras_in_section = [
            p for p in doc.paragraphs
            if this_start <= p.spans[0].start < next_start
        ]
        paras_in_section.sort(key=lambda p: p.spans[0].start)

        # Use the section title as the dictionary key (or a placeholder if None)
        heading = sec_entity.text if sec_entity is not None else "<no heading>"
        # Store paragraph texts in a list
        result[heading] = [p.text for p in paras_in_section]

    return result

def is_start_of_materials_methods(text):
    MATERIALS_METHODS_TITLES = [
        r"materials\s*(and|&)?\s*methods",
        r"materials",
        r"methodology",
        r"experimental",
        r"experimental\s+(procedure[s]?|section[s]?)",
        r"methods",
    ]

    pattern = re.compile(
        r"^\s*(\d+\.?|\b[IVXLCDM]+\b\.?)?\s*.*?\b(" + "|".join(MATERIALS_METHODS_TITLES) + r")\b",
        re.IGNORECASE
    )

    return bool(pattern.search(text.strip()))


def is_end_of_materials_methods(text):
    stop_keywords = ["RESULTS", "DISCUSSION", "CONCLUSION", "ACKNOWLEDGMENTS"]
    upper_text = text.strip().upper()
    return any(keyword in upper_text for keyword in stop_keywords)

def process_pdf_papermage(recipe, doc_id, doc_path, log_path, csv_text_path, csv_sections_path):
    setup_logger(log_path)
    os.makedirs(os.path.dirname(csv_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_sections_path), exist_ok=True)
    print(f"saving logs to {log_path}")
    logger.info(f"Reading PDF for file: {doc_path} ...")

    doc = recipe.run(doc_path)
    section_paragraph = group_paragraphs_by_section(doc)
    
    sub_sections = defaultdict(list)
    save_subs_conent = False
    current_title = None
    
    for section_title, section_content in section_paragraph.items():
        # section_content is a list of paragraph‐strings
        if is_start_of_materials_methods(section_title):
            save_subs_conent = True
            current_title = section_title
            print(f"*** Detected section start: '{section_title}'")
            # extend, not append, so each paragraph text goes in the list
            sub_sections[current_title].extend(section_content)
            continue

        elif save_subs_conent:
            if is_end_of_materials_methods(section_title):
                save_subs_conent = False
                print(f"*** Stopping at section: '{section_title}'")
                continue

            if current_title != section_title:
                print(f"*** Subsection: '{section_title}'")
            current_title = section_title

        if save_subs_conent:
            # again: section_content is a list, so extend
            sub_sections[current_title].extend(section_content)

    # Create section‐wise and full inline text
    section_rows = []
    final_text_parts = []

    for subsection in sorted(sub_sections):
        # sub_sections[subsection] is now a flat list of strings
        joined = " ".join(sub_sections[subsection])

        # Skip very short headings or empty text
        if not subsection or len(joined) < 5 or subsection.islower():
            continue

        section_rows.append({
            "doc_id": doc_id,
            "Subsection": subsection,
            "Text": joined
        })
        final_text_parts.append(f"{subsection}\n{joined}")

    final_text_string = "\n\n".join(final_text_parts)

    df_sections = pd.DataFrame(section_rows)
    df_final = pd.DataFrame([{
        "doc_id": doc_id,
        "Text": final_text_string
    }])

    df_final.to_csv(csv_text_path, index=False)
    df_sections.to_csv(csv_sections_path, index=False)

    return df_final, df_sections

def main():
    with zipfile.ZipFile("./cadmus/output_UoZ/retrieved_df/retrieved_df2.json.zip", "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                data = json.loads(data)

    f.close()
    z.close()
    metadata_retrieved_df = pd.read_json(data, orient='index')
    metadata_retrieved_df.pmid = metadata_retrieved_df.pmid.astype(str)
    cadmus_index_to_pmid = metadata_retrieved_df["pmid"].to_dict()
    
    # Initialize the PaperMage recipe
    recipe = CoreRecipe()

    # Path to folder containing PDFs
    pdf_folder = "./unzipped_pdfs_for_parsing_test/"

    # Output base paths
    base_log_path = "./unzipped_pdfs_for_parsing_test/materials_methods_papermage.log"
    output_base_path = "./unzipped_pdfs_for_parsing_test/papermage"

    # Make sure output directory exists
    os.makedirs(output_base_path, exist_ok=True)

    start_time = time.time()
    # Iterate through each PDF in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(pdf_folder, filename)
            doc_index = os.path.splitext(filename)[0]  # remove ".pdf"
            
            if doc_index not in cadmus_index_to_pmid:
                print(f"Skipping {filename}: index not found in mapping")
                continue

            doc_id = cadmus_index_to_pmid[doc_index]
            doc_output_folder = os.path.join(output_base_path, doc_index)
            os.makedirs(doc_output_folder, exist_ok=True)

            csv_text_path = os.path.join(doc_output_folder, f"{doc_id}_final_text.csv")
            csv_sections_path = os.path.join(doc_output_folder, f"{doc_id}_sections_text.csv")

            print(f"Processing {filename} as PMID {doc_id}")
            
            try:
                df_final, df_sections = process_pdf_papermage(
                    recipe,
                    doc_path=doc_path,
                    doc_id=doc_id,
                    log_path=base_log_path,
                    csv_text_path=csv_text_path,
                    csv_sections_path=csv_sections_path
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"Finished processing all PDFs in {minutes} minutes and {seconds} seconds.")