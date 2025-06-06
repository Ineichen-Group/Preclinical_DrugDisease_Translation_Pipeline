from cadmus import display_export_path
from cadmus import bioscraping
from cadmus import parsed_to_df
import zipfile
import json
import pandas as pd
import random 

#pmid_list = pd.read_csv("cadmus/filtered_df_non_empty_sclerosis_10043_PMIDs.csv")['PMID'].astype(str).tolist()
with open("cadmus/pmids_without_content_or_only_plain_UoZ.txt", "r") as f1:
    pmid_list = [line.strip() for line in f1 if line.strip()]
   
print(f"Will be fetching {len(pmid_list)} PMIDs.")

bioscraping(pmid_list,
            "donevasimona@gmail.com", #You need to insert your email address here
            "c73019fe6b2bfac722029994bf58e17a6d08", #You need to insert your NCBI_API_KEY here
            wiley_api_key = "867c3eec-80d0-413a-ac62-f2b752d84f71", #This is an optional parameter.
            #You can insert your WILEY_API_KEY here
            elsevier_api_key = "bd6a6874047737b559b59bd223673bae", #This is an optional parameter.
            #You can insert your ELSEVIER_API_KEY here
            )
