import pandas as pd
import csv
import re

def add_variant(canonical_name, variant, variant_to_canonical):
    """
    Associates a variant name (synonym, alternate name, or spelling variation) with a canonical name 
    (the official name) in the variant_to_canonical dictionary. If the variant does not already exist in 
    the dictionary, it is added as a key, with its value being a set that contains the canonical name.

    Parameters:
    - canonical_name (str): The official name or standardized form of the drug or condition.
    - variant (str): The synonym or alternate name that will be associated with the canonical name.
    - variant_to_canonical (dict): A dictionary mapping variants to sets of canonical names.

    Returns:
    - dict: The updated variant_to_canonical dictionary.
    """
    # Add variant if not already present, then add canonical name to the set
    variant_to_canonical.setdefault(variant, set()).add(canonical_name)
    return variant_to_canonical

def process_synonyms(synonyms_list, canonical_name, synonyms_dict):
    """
    Processes a list of synonyms and adds each synonym to the dictionary 
    using the add_variant function.
    
    Parameters:
    - synonyms_list (list): List of synonym strings.
    - canonical_name (str): The canonical name to associate with each synonym.
    - synonyms_dict (dict): The dictionary where synonyms are being stored.
    
    Returns:
    - dict: The updated synonyms dictionary.
    """
    canonical_name = canonical_name.lower().strip()
    for synonym in synonyms_list:
        synonym = synonym.lower().strip()
        synonyms_dict = add_variant(canonical_name, synonym, synonyms_dict)
    return synonyms_dict

def generate_conditions_lookup_dictionary(term_file_path):
    """
    Reads a CSV file containing condition names and synonyms, and generates a lookup dictionary that 
    maps each synonym (variant) to its canonical name (MeSH common name or ICD title).

    The CSV is expected to have the following columns:
    - 'ICD Title': The standardized ICD condition name.
    - 'MeSH Common name': The standardized MeSH condition name.
    - 'MeSH Synonyms': A list of synonyms for the MeSH condition name, separated by a pipe (|) character.

    Parameters:
    - term_file_path (str): The file path to the CSV containing the condition terms.

    Returns:
    - dict: A dictionary where keys are synonyms (variants) and values are sets of canonical names.
    """
    synonyms_dict = {}  # Initialize an empty dictionary to store variants and canonical names
    df = pd.read_csv(term_file_path)  # Read the CSV file into a DataFrame

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract values and make sure they're properly lowercased
        icd_title = row['ICD Title']
        mesh_name = row['MeSH Common name']
        mesh_synonyms = row['MeSH Synonyms']

        # If MeSH synonyms are present, process them
        if pd.notna(mesh_synonyms):
            synonyms_list = mesh_synonyms.split('|')
            synonyms_dict = process_synonyms(synonyms_list, mesh_name, synonyms_dict)
        
        # If there are no MeSH synonyms, use ICD title if available
        if pd.notna(icd_title):
            icd_title = icd_title.lower().strip()
            synonyms_dict = add_variant(icd_title, icd_title, synonyms_dict)
        
        # If neither of the above is valid, use the MeSH common name directly
        elif pd.notna(mesh_name):
            mesh_name = mesh_name.lower().strip()
            synonyms_dict = add_variant(mesh_name, mesh_name, synonyms_dict)

    return synonyms_dict  # Return the lookup dictionary
   
def add_drug(id, synonyms, drug_variant_to_canonical, drug_canonical_to_data, exclusions=None):
    """
    Add drug data to the lookup dictionaries.
    
    Parameters:
    - id (str): Drug identifier (e.g., medline ID, NHS URL, etc.).
    - synonyms (list): List of synonyms for the drug.
    - drug_variant_to_canonical (dict): The lookup dictionary where variants are mapped to canonical names.
    - drug_canonical_to_data (dict): The dictionary containing drug information for each canonical name.
    - exclusions (set, optional): A set of exclusion names to filter out certain drug names.
    """
    synonyms = [s.strip().lower() for s in synonyms]
    synonyms = [item for item in synonyms if item != '']

    if len(synonyms) == 0:
        return

    # Skip if the first synonym is in the exclusion list
    if exclusions and re.sub("[- ].+", "", synonyms[0].upper()) in exclusions:
        return
    
    # Initialize canonical entry if not already present
    if synonyms[0] not in drug_canonical_to_data:
        drug_canonical_to_data[synonyms[0]] = {"name": synonyms[0], "synonyms": set()}
        
    # Add the appropriate ID field to the canonical data
    if id.startswith("a"):
        drug_canonical_to_data[synonyms[0]]["medline_plus_id"] = id
    elif id.startswith("https://www.nhs.uk"):
        drug_canonical_to_data[synonyms[0]]["nhs_url"] = id
    elif id.startswith("https://en.wikipedia"):
        drug_canonical_to_data[synonyms[0]]["wikipedia_url"] = id
    elif id.startswith("DB"):
        drug_canonical_to_data[synonyms[0]]["drugbank_id"] = id
    else:
        drug_canonical_to_data[synonyms[0]]["mesh_id"] = id

    # Add each synonym and associate it with the canonical name
    for variant in synonyms:
        drug_canonical_to_data[synonyms[0]]["synonyms"].add(variant)
        add_variant(synonyms[0], variant.lower(), drug_variant_to_canonical)

def process_drug_file(file_path, delimiter, id_col, name_col, synonym_col, drug_variant_to_canonical, drug_canonical_to_data, exclusions=None):
    """
    Process a drug terminology file and add its content to the lookup dictionaries.
    
    Parameters:
    - file_path (str): The path to the drug file.
    - delimiter (str): The delimiter used in the CSV file.
    - id_col (int): The index of the column containing the drug identifier.
    - name_col (int): The index of the column containing the drug name.
    - synonym_col (int): The index of the column containing the drug synonyms.
    - drug_variant_to_canonical (dict): The lookup dictionary where variants are mapped to canonical names.
    - drug_canonical_to_data (dict): The dictionary containing drug information for each canonical name.
    - exclusions (set, optional): A set of exclusion names to filter out certain drug names.
    """
    with open(file_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        headers = next(reader, None)  # Skip the header row
        
        for row in reader:
            id = row[id_col]
            name = re.sub(r" (Injection|Oral Inhalation|Transdermal|Ophthalmic|Topical|Vaginal Cream|Nasal Spray|Transdermal Patch|Rectal)", "", row[name_col].lower())
            synonyms = row[synonym_col].split(r"|")
            add_drug(id, [name] + synonyms, drug_variant_to_canonical, drug_canonical_to_data, exclusions)

def generate_drugs_lookup_dictionary(file_paths, delimiter=',', exclusions=None):
    """
    Generate a lookup dictionary for drugs using data from multiple CSV files.

    Parameters:
    - file_paths (list of dict): A list of dictionaries where each contains:
        - 'path': The file path to the CSV.
        - 'id_col': The index of the drug ID column.
        - 'name_col': The index of the drug name column.
        - 'synonym_col': The index of the synonym column.
        - 'delimiter': The delimiter for the file (default is ',').
    - delimiter (str, optional): The default delimiter for the files (if not specified per file).
    - exclusions (set, optional): A set of exclusion names to filter out certain drug names.

    Returns:
    - tuple: 
        - drug_variant_to_canonical (dict): A dictionary mapping drug variants to canonical names.
        - drug_canonical_to_data (dict): A dictionary containing detailed data for each canonical drug.
    """
    # Initialize lookup dictionaries within the function
    drug_variant_to_canonical = {}
    drug_canonical_to_data = {}

    for file_info in file_paths:
        file_path = file_info['path']
        id_col = file_info.get('id_col', 0)
        name_col = file_info.get('name_col', 1)
        synonym_col = file_info.get('synonym_col', 2)
        file_delimiter = file_info.get('delimiter', delimiter)
        
        process_drug_file(file_path, file_delimiter, id_col, name_col, synonym_col, drug_variant_to_canonical, drug_canonical_to_data, exclusions)
    
    return drug_variant_to_canonical, drug_canonical_to_data

def lookup_canonical(conditions_list, synonyms_dict):
    """
    Looks up the canonical form of conditions from a list and matches them to entries in a synonyms dictionary. 
    It also tracks statistics on the number of matches found and unchanged conditions.

    Parameters:
    - conditions_list (str): A pipe-separated string of conditions (variants).
    - synonyms_dict (dict): Dictionary mapping variants to canonical condition names.

    Returns:
    - tuple: 
        - (str) Canonical list joined by '|' character.
        - (int) Number of matches found.
        - (int) Number of processed conditions (non-skipped).
        - (int) Number of conditions that remain unchanged.
    """
    if not isinstance(conditions_list, str):
        return '', 0, 0, 0  # Return default values if input is invalid

    canonical_list = []
    match_count = 0
    processed_count = 0
    same_condition_count = 0

    for condition in conditions_list.split('|'):
        original_condition = condition  # Store the original condition
        condition = condition.lower().strip()

        # Skip if condition is irrelevant (empty, none, etc.)
        if condition in {"none", "", "none."}:
            continue

        processed_count += 1  # Increment for every non-skipped condition

        # Look up condition in the synonyms dictionary
        if condition in synonyms_dict:
            canonical_list.extend(synonyms_dict[condition])
            match_count += 1  # Increment when a match is found
            
            # Check if the original condition is already in its canonical form
            if original_condition in synonyms_dict[condition]:
                same_condition_count += 1
        else:
            canonical_list.append(condition)  # If no match, keep the original condition

    if len(canonical_list) > 1:
        canonical_list_str = '|'.join(canonical_list)
    else:
        canonical_list_str = ''.join(canonical_list)
    return canonical_list_str, match_count, processed_count, same_condition_count

def process_dataframe(df, synonyms_dict, source_col, entity_type="conditions"):
    """
    Processes a DataFrame to map conditions from a source column to their canonical forms using a synonyms dictionary. 
    It also tracks and returns statistics on the number of matches, processed conditions, and unchanged conditions.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.
    - synonyms_dict (dict): The dictionary that maps condition variants to canonical names.
    - source_col (str): The column in the DataFrame that contains condition strings to be mapped.
    - entity_type (str): A string to define the type of entity being mapped (default is "conditions").

    Returns:
    - tuple:
        - (pd.DataFrame): The updated DataFrame with canonical conditions in a new column.
        - (dict): Match counts for the newly created canonical column.
        - (dict): Processed counts for the canonical column.
        - (dict): Counts where the condition remained unchanged for the canonical column.
    """
    # Initialize dictionaries to track counts
    match_counts = {}
    processed_counts = {}
    same_condition_counts = {}

    # Define the name for the new column based on entity type
    column = f'linkbert_mapped_{entity_type}'

    # Apply the lookup function to each entry in the specified column
    results = df[source_col].apply(lambda x: lookup_canonical(x, synonyms_dict))

    # Extract the results from the tuple returned by lookup_canonical
    df[column] = [result[0] for result in results]  # Populate the new column with canonical forms
    match_counts[column] = sum(result[1] for result in results)  # Total number of matches
    processed_counts[column] = sum(result[2] for result in results)  # Total number of processed conditions
    same_condition_counts[column] = sum(result[3] for result in results)  # Total number of unchanged conditions

    return df, match_counts, processed_counts, same_condition_counts