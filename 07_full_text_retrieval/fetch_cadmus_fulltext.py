from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import json
import random 

def load_api_keys(filepath):
    api_keys = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue
            
            key, value = line.split("=", 1)
            api_keys[key.strip()] = value.strip()
    
    return api_keys

path_to_pmids="no_methods_pmids.txt"

with open(path_to_pmids, "r") as f1: #, open("failed_pmc_pmids.txt", "r") as f2:
    pmid_list_1 = [line.strip() for line in f1 if line.strip()]
    #pmid_list_2 = [line.strip() for line in f2 if line.strip()]

# Combine and optionally remove duplicates
pmid_list = list(set(pmid_list_1)) # + pmid_list_2))

print(f"Will be fetching {len(pmid_list)} PMIDs.")

keys = load_api_keys("api_keys.txt")

wiley_api_key_uoz = keys.get("wiley_api_key_uoz")
elsevier_api_key_uoz = keys.get("elsevier_api_key_uoz")
ncbi_api_key = keys.get("NCBI_API_KEY")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            ncbi_api_key, #You need to insert your NCBI_API_KEY here
            wiley_api_key = wiley_api_key_uoz, #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = elsevier_api_key_uoz, #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
