# Preclinical_Pipeline

## PubMed Query

### Setup
We used the EDirect package, which includes several commands that use the E-utilities API to find and retrieve PubMed data. You can install it via the command:
```
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
```

Note:  For best performance, obtain an API Key from NCBI, and place the following line in your .bash_profile and .zshrc configuration files (follow https://support.nlm.nih.gov/kbArticle/?pn=KA-05317):
```
  export NCBI_API_KEY=unique_api_key
```

Relevant API documentation references:

- https://www.ncbi.nlm.nih.gov/books/NBK179288/
- https://www.nlm.nih.gov/dataguide/edirect/xtract_formatting.html
- https://dataguide.nlm.nih.gov/classes/edirect-for-pubmed/samplecode4.html#specify-a-placeholder-to-replace-blank-spaces-in-the-output-table

### From query to PMIDs list
To obtain the initial set of relevant PMIDs, the database was queried using several search string related to CNS and Psychiatric conditions, as follows:
- [./01_pubmed_query_neuro/PubMed_Query_Prep.ipynb](./01_pubmed_query_neuro/PubMed_Query_Prep.ipynb) - we first split the very long CNS related free text search queries into smaller chunks
- The chunks are found in [./01_pubmed_query_neuro/data/pubmed_queries/nervous_system/](./01_pubmed_query_neuro/data/pubmed_queries/nervous_system/). Those were executed iteratively with the script [./scripts/pubmed_query_long.sh](./01_pubmed_query_neuro/scripts/pubmed_query_long.sh). 
In addition MeSH based queries are in [./1_pubmed_query_neuro/data/pubmed_queries/nervous_system/cns_mesh_1.txt](./01_pubmed_query_neuro/data/pubmed_queries/nervous_system/cns_mesh_1.txt) and [./01_pubmed_query_neuro/data/pubmed_queries/nervous_system/cns_mesh_2.txt](./01_pubmed_query_neuro/data/pubmed_queries/nervous_system/cns_mesh_2.txt)

- Psychiatric conditions related queries: see [./01_pubmed_query_neuro/data/pubmed_queries/psychiatric/psychiatrich_query_used.txt](./01_pubmed_query_neuro/data/pubmed_queries/psychiatric/psychiatrich_query_used.txt).


The queries use the following syntax:
```
esearch -db pubmed -query '(Central nervous system diseases[MeSH] OR Mental Disorders OR Psychiatric illness[MeSH]) AND English[lang]' | efetch -format uid > "./cns_psychiatric_diseases_pmids_en_$(date +%Y%m%d).txt"
```

The outputs from those queries were stored in [./01_pubmed_query_neuro/data/pubmed_queries/results_pmids/](./01_pubmed_query_neuro/data/pubmed_queries/results_pmids/). They were then combined into a full list of PMIDs in [./01_pubmed_query_neuro/PubMed_Query_PMIDs_List.ipynb](./01_pubmed_query_neuro/PubMed_Query_PMIDs_List.ipynb).

### From PMIDs to content
1. Split into chunks of 5000 PMIDs per file, see [./01_pubmed_query_neuro/scripts/server/split_pmids_to_chunks.sh](./01_pubmed_query_neuro//scripts/split_pmids_to_chunks.sh)

2. Run parallel fetching of content on the surver for each chunk, see [./01_pubmed_query_neuro/scripts/server/fetch_pubmed_data_large_no_doi_with_retry.sh](./01_pubmed_query_neuro/scripts/fetch_pubmed_data_large_no_doi_with_retry.sh)

This executes the following call:
```
efetch -db pubmed -id "$id_list" -format xml 2>> error.log | \
        xtract \
          -pattern PubmedArticle -tab "|||" -def "N/A" \
          -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
          -block PublicationTypeList -sep "+" -element PublicationType \
        > "$OUTPUT_FILE"
```

The script includes retries in case the API call fails.

However, there was always an incomplete set of returned results. Therefore after each iteration, we obtained the PMIDs for which no content was present, and re-ran the steps 1-3. Future work is to automate this process even further. The data is saved in [./01_pubmed_query_neuro/data/full_pubmed_raw/](./01_pubmed_query_neuro/data/full_pubmed_raw/). There are still some publications that kept failing, this can be checked with [./01_pubmed_query_neuro/check_failed_queries.py](./01_pubmed_query_neuro/check_failed_queries.py).


## Animal Studies Classification
The StudyTypeTeller full dataset is [./02_animal_study_classification/data/full_enriched_dataset_2696.csv](./02_animal_study_classification/data/full_enriched_dataset_2696.csv).

### Finetuning
1. Prep data for fine tuning. We split it into 90-10% for training and validation using the script [./02_animal_study_classification/generate_stratified_splits.py](./02_animal_study_classification/generate_stratified_splits.py).
2. Fine tune a binary classifier. The we use this split to fine-tune a binary SciBERT classifier, see [./02_animal_study_classification/run_finetune.sh](./02_animal_study_classification/run_finetune.sh).

### Inference 

3. Perform inference over the fetched PubMed abstracts. In the script [./02_animal_study_classification/run_parallel_inference.sh](./02_animal_study_classification/run_parallel_inference.sh), we point to the folder where the PubMed contents are stored from the previous step. Those are different chunks from the parallel fetching of PubMed contents. We parallelize the inference to predict the study type (animal vs other) for each publication. The predictions are stored to [./02_animal_study_classification/model_predictions/](./02_animal_study_classification/model_predictions/).
4. To quickly extract the PMIDs only of animal studies we used the command below, resulting in ./[02_animal_study_classification/model_predictions/all_animal_studies.txt](02_animal_study_classification/model_predictions/all_animal_studies.txt):
   ```
   find pubmed_results* -type f -name "*.txt" -exec grep 'Animal' {} + > all_animal_studies.txt
   ```
5. In a next step the full corpus of publications is filtered for the animal studies. This happens in [./02_animal_study_classification/filter_preclinical_pmids_save_for_ner.py](./02_animal_study_classification/filter_preclinical_pmids_save_for_ner.py). 
6. In the same script the animal studies are also split into chunks that will be used for inference for NER.

## NER 

### Finetuning 

### Inference 
The animal studies were prepared for NER inference by splitting them into chunks of PMID and Text.
The inference code to run on the server over those chunks in parallel is [./03_ner/run_ner_inference_animal.sh](./03_ner/run_ner_inference_animal.sh). This calls [./03_ner/inference_ner_annotations.py](./03_ner/inference_ner_annotations.py), which in turn works with an NERModel from [./03_ner/core/models.py](./03_ner/core/models.py).


### Post-processing 
Post processing of the obtained NER predictions happens in [./03_ner/process_ner_predictions.py](./03_ner/process_ner_predictions.py). This includes:
1. Load NER Predictions  
   Reads NER model outputs (predicted entities) and excludes the articles have no predictions at all.

2. Abbreviation Expansion  
   For each article, if not already cached, the script extracts abbreviation-definition pairs from the full text to aid in resolving entity meanings. Save in [./03_ner/data/abbreviations_expansion/pmid_abbreviations.csv](./03_ner/data/abbreviations_expansion/pmid_abbreviations.csv).

3. Merge Predictions with Abbreviations  
   Joins abbreviation data with NER predictions by article (PMID).

4. Extract Unique Entities  
   For each article, extracts a unique list of predicted conditions and interventions, using both the NER predictions and abbreviation expansions.

5. Filter Articles  
   Keeps only those articles that mention at least one condition and one intervention.

6. Save Filtered Data  
   Saves in [./03_ner/data/animal_studies_with_drug_disease/](./03_ner/data/animal_studies_with_drug_disease/):
   - All qualifying articles with both entity types.
   - The corresponding PMIDs.
   - A subset of articles mentioning “sclerosis”, for focused study.

**Outputs:**

All saved files are written to the `03_ner/data/animal_studies_with_drug_disease/` directory and include:

- Filtered articles with both condition and intervention mentions.
- PMIDs of those articles.
- Sclerosis-specific subset.


## Validation with existing systematic reviews 

We took the following systematic reviews:
- [./04_syst_reviews_validation/data/Hyperoside.pdf](./04_syst_reviews_validation/data/Hyperoside.pdf)
- [./04_syst_reviews_validation/data/Lithium.pdf](./04_syst_reviews_validation/data/Lithium.pdf)
- [./04_syst_reviews_validation/data/Masitinib.pdf](./04_syst_reviews_validation/data/Masitinib.pdf)

And identified the expected animal article PMIDs saved in [./04_syst_reviews_validation/data/PMID.xlsx](./04_syst_reviews_validation/data/PMID.xlsx).

We check those target PMIDs are present in our dataset with the same drug and disease mentions. The code for that is in [./04_syst_reviews_validation/sys_review_validation.py](./04_syst_reviews_validation/sys_review_validation.py). 
This results in three files from the courpus filtered for the PMIDs of each systematic reviews and showing the unique extracted entities. 





