import pandas as pd

def read_pmids_from_excel(file_path):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Dictionary to store PMIDs for each sheet
    pmid_dict = {}
    
    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Assuming PMIDs are in the first column
        pmids = df.iloc[:, 0].tolist()
        
        # Store the list of PMIDs in the dictionary
        pmid_dict[sheet_name] = pmids
    
    return pmid_dict

# Example usage
file_path = "04_syst_reviews_validation/data/PMID.xlsx"
pmid_dict = read_pmids_from_excel(file_path)

# All studies returned from the pubmed query
all_original_pmids_file = "01_pubmed_query_neuro/pmids_with_data.txt"
with open(all_original_pmids_file) as f:
    all_query_pmids = set(map(int, filter(str.isdigit, f.read().splitlines())))
    
# Predicted animal studies
all_animal_studies = pd.read_csv("02_animal_study_classification/model_predictions/all_animal_studies_clean_complete.csv")

# All studies with drug and disease mention which will be kept for future processing
identified_studies = pd.read_csv("03_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_595768.csv")[['PMID', 'unique_interventions_linkbert_predictions', 'unique_conditions_linkbert_predictions']]

# Filter identified studies for each sheet
filtered_studies = {}
for sheet, pmids in pmid_dict.items():
    print('\n' + sheet)
    print(f"target pmids {len(pmids)}")
    # Check if all target PMIDs are in the identified PMIDs
    target_pmids = set(pmids)
    print("targets: ", target_pmids)
    
        # Filter identified studies for the rows with target_pmids
    matched_pmids = identified_studies[identified_studies['PMID'].isin(target_pmids)]
    matched_pmids.to_csv(f"./04_syst_reviews_validation/{sheet}_predicted.csv")
    
    # Filter identified studies for the current intervention
    filtered_studies[sheet] = identified_studies[identified_studies['unique_interventions_linkbert_predictions'].str.lower().str.contains(sheet, case=False)]
    
    print(f"identified pmids {len(filtered_studies[sheet])}")

    identified_pmids = set(filtered_studies[sheet]['PMID'])
    missing_pmids = target_pmids - identified_pmids
    if len(missing_pmids) == 0:
        print("All target PMIDs are in the identified PMIDs")
    else:
        print("Not all target PMIDs are in the identified PMIDs")
        print("Missing PMIDs:", missing_pmids)
        
        # IS IT IN ORIGINAL set of studies?
        original_query_pmids = set(all_query_pmids)
        missing_pmids = target_pmids - original_query_pmids
        if len(missing_pmids) == 0:
            print("All target PMIDs are in the original set of PMIDs")
        else:
            print("Not all target PMIDs are in the original PMIDs list")
            print("Missing PMIDs:", missing_pmids)
            
        # IS IT IN ANIMAL studies?
        animal_pmids = set(all_animal_studies['PMID'])
        missing_pmids = target_pmids - animal_pmids
        if len(missing_pmids) == 0:
            print("All target PMIDs are in the animal set of PMIDs")
        else:
            print("Not all target PMIDs are in animal PMIDs list")
            print("Missing PMIDs:", missing_pmids)
        
        # IS IT IN NER identified?
        animal_ner_pmids = set(identified_studies['PMID'])
        missing_pmids = target_pmids - animal_pmids
        if len(missing_pmids) == 0:
            print("All target PMIDs are in animal with NER set of PMIDs")
        else:
            print("Not all target PMIDs are in the animal with NER PMIDs list")
            print("Missing PMIDs:", missing_pmids)
        


