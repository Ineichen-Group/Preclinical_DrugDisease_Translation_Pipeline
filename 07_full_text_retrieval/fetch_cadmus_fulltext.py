from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import json
import random 

path_to_pmids="no_methods_pmids.txt"

with open(path_to_pmids, "r") as f1: #, open("failed_pmc_pmids.txt", "r") as f2:
    pmid_list_1 = [line.strip() for line in f1 if line.strip()]
    #pmid_list_2 = [line.strip() for line in f2 if line.strip()]

# Combine and optionally remove duplicates
pmid_list = list(set(pmid_list_1)) # + pmid_list_2))

#random.seed(42)  # Set a seed value for the random number generator
#pmid_list = random.sample(pmid_list, k=300)
    
print(f"Will be fetching {len(pmid_list)} PMIDs.")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            "c73019fe6b2bfac722029994bf58e17a6d08", #You need to insert your NCBI_API_KEY here
            wiley_api_key = "a59e4084-c74d-466c-9985-88128de4e13f", #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = "034e0b4ff44a1f96f08bf85559dc9eaa", #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
