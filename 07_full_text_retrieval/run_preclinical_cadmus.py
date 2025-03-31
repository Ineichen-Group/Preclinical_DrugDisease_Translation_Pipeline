from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import zipfile
import json
import pandas as pd
import random 

pmid_list = pd.read_csv("filtered_df_non_empty_sclerosis_10043_PMIDs.csv")['PMID'].astype(str).tolist()
random.seed(42)  # Set a seed value for the random number generator
pmid_list = random.sample(pmid_list, k=300)
    
print(f"Will be fetching {len(pmid_list)} PMIDs.")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            "c73019fe6b2bfac722029994bf58e17a6d08", #You need to insert your NCBI_API_KEY here
            wiley_api_key = "f11ecc1a-4b5e-4316-bd73-926423703485", #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = "034e0b4ff44a1f96f08bf85559dc9eaa", #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
