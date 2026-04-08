from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import zipfile
import json
import pandas as pd
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

#pmid_list = pd.read_csv("cadmus/filtered_df_non_empty_sclerosis_10043_PMIDs.csv")['PMID'].astype(str).tolist()
with open("cadmus/pmids_without_content_or_only_plain_UoZ.txt", "r") as f1:
    pmid_list = [line.strip() for line in f1 if line.strip()]
   
print(f"Will be fetching {len(pmid_list)} PMIDs.")

keys = load_api_keys("api_keys.txt")


ncbi_api_key = keys.get("NCBI_API_KEY")
wiley_api_key_uoe = keys.get("wiley_api_key_uoe")
elsevier_api_key_uoe = keys.get("elsevier_api_key_uoe")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            ncbi_api_key, #You need to insert your NCBI_API_KEY here
            wiley_api_key = wiley_api_key_uoe, #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = elsevier_api_key_uoe, #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
