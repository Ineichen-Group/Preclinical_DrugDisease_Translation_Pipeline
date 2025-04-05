import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def read_df_from_txt(input_file_path, headers, separator, target_pmids):
    """
    Reads a large text file line by line and filters rows based on target PMIDs.
    """
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split(separator)
            if len(fields) == len(headers):  # Ensure correct number of fields
                row = dict(zip(headers, fields))
                if target_pmids and (int(row["PMID"]) in target_pmids):  # Filter by PMID
                    yield row
                elif not target_pmids:
                    yield row
                    

def extract_pmids_from_file(file_path, separator="|||"):
    """
    Reads a file and extracts the first column (PMID) as a set.
    """
    pmid_set = set()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split(separator)
            if fields:  # Ensure line is not empty
                pmid_set.add(fields[0])  # Assuming PMID is the first column
    
    return pmid_set

def load_folder_files(files_path, headers, separator='|||', target_pmids=None):
    """
    Reads and filters multiple large files in a folder efficiently.
    """
    file_list = glob.glob(os.path.join(files_path, '*.txt'))
    
    
    for file_path in file_list:
        if os.path.getsize(file_path) == 0:
            continue  # Skip empty files
        fetched_pmids = extract_pmids_from_file(file_path)
        
        chunk_data = []
        for row in read_df_from_txt(file_path, headers, separator, target_pmids):
            row['abstract_missing'] = row['abstract'] in [None, "", "N/A"]
            row['chunk'] = os.path.basename(file_path)
            chunk_data.append(row)

            if len(chunk_data) >= 100000:  # Process in chunks
                yield pd.DataFrame(chunk_data)
                chunk_data.clear()

        if chunk_data:  # Process remaining data
            yield pd.DataFrame(chunk_data)

def load_pubmed_raw_filter_animal_save_for_ner(pubmed_content_path, output_path, target_pmids=None):
    """
    Loads and filters large text files based on given PMIDs and writes to a CSV in chunks.
    """
    headers = ["PMID", "year", "journal_name", "title", "abstract", "publication_type"]
    separator = '|||'
    
    data_folders = [
        'missing_chunks_data', 'round_1', 'round_2', 'round_3', 'round_4',
        'round_5', 'round_6', 'round_7', 'round_8'
    ]

    if target_pmids:
        print(f'Filtering for {len(target_pmids)} PMIDs')
        output_file_base = f"{output_path}/full_pubmed_filtered_animal_{len(target_pmids)}"
    else:
        output_file_base = f"{output_path}/full_pubmed"

    first_write = True  # Track first chunk write
    total_rows = 0
    missing_abstract_rows = 0

    for folder in data_folders:
        folder_path = f'{pubmed_content_path}/{folder}'
        print(f'Processing {folder_path}')
        count_chunk = 0 
        for df_chunk in load_folder_files(folder_path, headers, separator, target_pmids):
            #print(f'chunk size {df_chunk.shape}')
            df_chunk.drop_duplicates(subset="PMID", inplace=True)
            
            # Save METADATA
            df_metadata = df_chunk.copy()[['PMID', 'year', 'journal_name','publication_type', 'title']]
            df_metadata.to_csv(f'{output_file_base}_metadata.csv', mode='w' if first_write else 'a', header=first_write, index=False)
            
            # Save in format for NER
            df_chunk['Text'] = df_chunk['title'] + " | " + df_chunk['abstract']
            df_for_ner = df_chunk.copy()[['PMID', 'Text']]
            saved_file_name_ner = f'{output_file_base}_for_NER.csv'
            df_for_ner.to_csv(saved_file_name_ner, mode='w' if first_write else 'a', header=first_write, index=False)
            
            # Count and save rows with missing abstract
            missing_abstract_rows += df_chunk['abstract_missing'].sum()
            df_missing_abstract = df_chunk[df_chunk['abstract_missing']]
            df_missing_abstract.to_csv(f'{output_file_base}_missing_abstract.csv', mode='w' if first_write else 'a', header=first_write, index=False)
                        
            first_write = False  # Switch to append mode
            total_rows += len(df_chunk)
            count_chunk += 1

    print(f'Loaded {total_rows} rows matching the given PMIDs')
    print(f'Number of rows with missing abstract: {missing_abstract_rows}')
    return saved_file_name_ner

def load_and_clean_study_type_teller_animal(file_path, output_file):
 
    df = pd.read_csv(file_path, names=["PMID", "label_id", "label_name", "confidence"], header=None)

    # Remove any prefixes before the actual data (file paths)
    df["PMID"] = df["PMID"].astype(str).str.split(":").str[-1]

    print(f"animal studies saved {len(df)}")
    # Save the cleaned data to a CSV file
    df.to_csv(output_file, index=False)
    
        
def split_ner_data_to_chunks(saved_file_name_ner, output_dir_ner_chunks):
    print(f"Splitting animal studies for NER inference from: {saved_file_name_ner}")
    input_file = saved_file_name_ner  # Original large file
    chunk_size = 10_000  # Number of rows per chunk
    output_dir = output_dir_ner_chunks # Directory to store split files

    # Read and save in chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        output_chunk_file = os.path.join(output_dir, f'pubmed_filtered_animal_for_NER_chunk_{i+1}.csv')
        chunk.to_csv(output_chunk_file, index=False)  # Save each chunk as a new file

    print(f"All chunks saved successfully to {output_dir_ner_chunks}.")
    
def main():
    # Clean Animal studies classificatin output
    file_path = "./02_animal_study_classification/model_predictions/all_animal_studies_complete.txt"
    output_file_animal_studies = "./02_animal_study_classification/model_predictions/all_animal_studies_clean_complete.csv"
    #load_and_clean_study_type_teller_animal(file_path, output_file_animal_studies)
    
    # Filter the fetched PMID contents for the Animal studies
    animal_studies = pd.read_csv(output_file_animal_studies)[['PMID']]
    animal_pmids = set(map(int, list(animal_studies['PMID'])))
    pubmed_content_path = "./01_pubmed_query_neuro/data/full_pubmed_raw"
    filtered_output_path = "./02_animal_study_classification/data/animal_studies"
    saved_file_name_ner = load_pubmed_raw_filter_animal_save_for_ner(pubmed_content_path, filtered_output_path, animal_pmids)
    
    output_dir_ner_chunks = "02_animal_study_classification/data/animal_studies_for_ner"
    #split_ner_data_to_chunks(saved_file_name_ner, output_dir_ner_chunks)

if __name__ == "__main__":
    main()