from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import zipfile
import json
import pandas as pd
import random 

#pmid_list = pd.read_csv("filtered_df_non_empty_sclerosis_10043_PMIDs.csv")['PMID'].astype(str).tolist()
with open("failed_pmc_pmids.txt", "r") as f:
    pmid_list = [line.strip() for line in f if line.strip()]

random.seed(42)  # Set a seed value for the random number generator
pmid_list = random.sample(pmid_list, k=300)
    
print(f"Will be fetching {len(pmid_list)} PMIDs.")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            "c73019fe6b2bfac722029994bf58e17a6d08", #You need to insert your NCBI_API_KEY here
            wiley_api_key = "XXX-XXX-XXX", #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = "XXX-XXX-XXX", #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
