import glob
import pandas as pd
import os

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

def save_pmids_with_data(files_path, output_pmids_with_content, separator='|||', target_pmids=None):
    """
    Reads and filters multiple large files in a folder efficiently.
    """
    if os.path.exists(output_pmids_with_content):
        print(f"Warning: {output_pmids_with_content} already exists. Stopping.")
        return
    
    data_folders = [
        'round_1', 'round_2', 'round_3', 'round_4',
        'round_5', 'round_6', 'round_7', 'round_8'
    ]
    file_list = []
    for folder in data_folders:
        print(f"Processing {folder}")
        folder_path = os.path.join(files_path, folder)
        file_list.extend(glob.glob(os.path.join(folder_path, '*.txt')))
    
    for file_path in file_list:
        if os.path.getsize(file_path) == 0:
            continue  # Skip empty files
        fetched_pmids = extract_pmids_from_file(file_path)
    
        # Save 
        with open(output_pmids_with_content, 'a', encoding='utf-8') as out_file:
            for pmid in sorted(fetched_pmids):  # Sort to maintain order
                out_file.write(f"{pmid}\n")
       
       
def check_pmids_missing_data(pmids_with_content_file, all_expected_pmids_file):
    with open(pmids_with_content_file) as f:
        with_data_pmids = set(map(int, filter(str.isdigit, f.read().splitlines())))
        
    with open(all_expected_pmids_file) as f:
        all_target_pmids = set(map(int, filter(str.isdigit, f.read().splitlines())))

    # Read written data in chunks instead of loading it all
    missing_pmid = all_target_pmids - with_data_pmids

    print(f'Expected: {len(all_target_pmids)}, Found: {len(with_data_pmids)}, Missing: {len(missing_pmid)}')

    output_file_path = f'./01_pubmed_query_neuro/data/pubmed_queries/missing_pmids_{len(missing_pmid)}.txt'
    with open(output_file_path, 'w') as file:
        file.writelines(f"{pmid}\n" for pmid in missing_pmid)
        
def main():
    pubmed_content_path = "./01_pubmed_query_neuro/data/full_pubmed_raw/"
    output_pmids_with_content = "./01_pubmed_query_neuro/data/full_pubmed_raw/pmids_with_data.txt"
    save_pmids_with_data(pubmed_content_path, output_pmids_with_content)
    
    all_expected_pmids_file = "./01_pubmed_query_neuro/data/pubmed_queries/union_all_queries_pmids_21704840.txt"
    check_pmids_missing_data(output_pmids_with_content, all_expected_pmids_file)

if __name__ == "__main__":
    main()