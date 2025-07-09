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

## Named entity recognition (NER) 

### Finetuning 
The datasets for fine-tuning are prepared in [./03_IE_tasks/Prep_NER_data.ipynb](./03_IE_tasks/Prep_NER_data.ipynb). 

The fine-tuning code is in [./03_ner/train_bert_ner.py](./03_ner/train_bert_ner.py) and the script to run in parallel on the server for k-fold cross-validation is in
 [./03_ner/run_k_fold_parallel_experiment.sh](./03_ner/run_k_fold_parallel_experiment.sh).

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

## Named entity normalization (NEN) 
We first have a basic dictionary based mapping to map terms to a more standardized form. However this has very low coverage, and is sensitive to terms who diverge too much from the spelling in the dictionary. Code for this is in [./04_normalization/dictionary_based_nen.py](./04_normalization/dictionary_based_nen.py).

To further normalize the extracted entities we use two target sources:
1. Disease terms are mapped to MONDO, downloaded from [https://mondo.monarchinitiative.org/pages/download/](https://mondo.monarchinitiative.org/pages/download/).
2. Drug terms are mapped to UMLS. The "2024AB Full UMLS Release Files" was downloaded from [https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). It is filtered for levels 0 (public) and 9 (SNOMED), as well as selected biomedical DBs. Further it is filtered for the semantic types "Organic Chemical", "Clinical Drug", "Biologically Active Substance", "Amino Acid, Peptide, or Protein", "Enzyme", "Immunologic Factor", "Nucleic Acid, Nucleoside, or Nucleotide", "Inorganic Chemical", "Antibiotic", "Biomedical or Dental Material", "Hormone", "Element, Ion, or Isotope", "Drug Delivery Device", "Vitamin", "Chemical Viewed Structurally", "Chemical".

### MONDO/UMLS embedding via SapBERT
The script [./04_normalization/embed_ontology.py](./04_normalization/embed_ontology.py) generates vector embeddings for the ontology terms from MONDO and UMLS using the SapBERT model. 

The process involves:
- Loading ontology terms (from MONDO .owl files or UMLS .csv exports)
- Embedding term names using SapBERT
- Saving the resulting embeddings and term metadata for later use in normalization

**Files and Outputs**

MONDO:
- Input: mondo.owl
- Output: 
    - JSON file of (term name, MONDO ID) pairs
    - .npy files containing embeddings of MONDO terms

UMLS:
- Input: CSV file of filtered MRCONSO terms (e.g. drug/chemical concepts)
- Output:
    - JSON file of (term name, CUI) pairs
    - .npy files containing embeddings of UMLS terms
  
### Selection of Best cdist Parameter
We manually annotated 100 randomly sampled disease and drug NER entities (see folder [./04_normalization/data/ner_samples/](./04_normalization/data/ner_samples/)). Each entity was then mapped to its closest SapBERT embedding from the relevant ontologies, and the embedding distance was recorded.

Using these distances, we estimated precision and recall at various cosine distance (cdist) thresholds. Entities with a distance above the selected threshold were not mapped and instead returned as the original NER text. For implementation details, see [./04_normalization/estimate_cdist_tsh_parameter.py](./04_normalization/estimate_cdist_tsh_parameter.py).

### Normalization Script
The script [./04_normalization/neural_based_nen.py](./04_normalization/neural_based_nen.py) performs entity normalization by mapping raw condition mentions (from NER output) to standardized MONDO/UMLS ontology terms. It uses the precomputed embeddings (from SapBERT) to match input terms to the closest ontology concept.

The normalization process:
- Loads MONDO/ UMLS embeddings and term metadata
- Uses a pretrained SapBERT model to embed query terms
- Computes similarity between query embeddings and ontology embeddings
- Maps terms to MONDO/UMLS if similarity is above a defined threshold, otherwise keeps the original NER output
- Logs normalization statistics and saves normalized results

**Files and Outputs**
Input:
- A CSV file containing annotated mentions (e.g., from NER)
- Column with mapped via dictionary condition/ drug strings (e.g. `linkbert_mapped_conditions`)
- MONDO/ UMLS embeddings (.npy files) and term metadata (.json)

Output:
- A new CSV with normalized MONDO/ UML mappings per row
- A summary stats log file with mapping performance.
- Logs of successfully and failed mapping entities.
- Normalized fields include e.g.:
    - `linkbert_mondo_conditions`: MONDO concept names
    - `mondo_termid`: MONDO concept IDs
    - `mondo_term_norm`: Canonical forms
    - `mondo_closest_3`: Top 3 closest MONDO concepts
    - `mondo_cdist`: Embedding distance to closest concept

To run this script on the server with data parallelism for drug/disease see [./04_normalization/run_normalize_parallel.sh](./04_normalization/run_normalize_parallel.sh):
```
sbatch run_normalize_parallel.sh disease
```

## Validation 

### Validation with existing systematic reviews 

We took the following systematic reviews:
- [./04_syst_reviews_validation/data/Hyperoside.pdf](./04_syst_reviews_validation/data/Hyperoside.pdf)
- [./04_syst_reviews_validation/data/Lithium.pdf](./04_syst_reviews_validation/data/Lithium.pdf)
- [./04_syst_reviews_validation/data/Masitinib.pdf](./04_syst_reviews_validation/data/Masitinib.pdf)

And identified the expected animal article PMIDs saved in [./04_syst_reviews_validation/data/PMID.xlsx](./04_syst_reviews_validation/data/PMID.xlsx).

We check those target PMIDs are present in our dataset with the same drug and disease mentions. The code for that is in [./04_syst_reviews_validation/sys_review_validation.py](./04_syst_reviews_validation/sys_review_validation.py). 
This results in three files from the courpus filtered for the PMIDs of each systematic reviews and showing the unique extracted entities. 

### Validation with MS-SOLES 


## Full-text retrieval and methods extraction

### Full-text retrieval
PMC -> Cadmus

### Methods extraction
#### From PMC BioC fromat
The methods extraction from PMC BioC format is done in [./07_full_text_retrieval/extract_methods_from_bioc_json.py](./07_full_text_retrieval/extract_methods_from_bioc_json.py). It uses several strategies to extract the methods section from the full text BioC XML of the publications.


#### From Cadmus
All CADMUS outputs are processed by format-specific parsers in: [./07_full_text_retrieval/cadmus_extractors/](./07_full_text_retrieval/cadmus_extractors/). The parser are develed for:
- **XML** (`xml_extractor.py`)  
  Parses XML files, finds the “Materials and Methods” section (or JATS-style `<sec>`), and outputs `methods_subtitles_<PMID>.csv`
- **HTML** (`html_extractor.py`)  
Loads HTML ZIPs, applies several strategies (e.g. `<section data-title="…">`, `<div id="methods">`, OVID/NLM layouts, etc.) to locate M&M text, and writes `methods_subtitles_<PMID>.csv`.
- **PDF** (`pdf_extractor.py`)  
Uses PaperMage to extract section headings and paragraphs. Looks for the first “Materials and Methods” heading, collects its text and subsections, then writes two CSVs under the output folder:
  - `<PMID>_sections.csv` (one row per subsection)
  - `<PMID>_full_text.csv` (all M&M text in one row)
- **Plain Text** (`plain_extractor.py`)  
Reads `.txt` files, uses a regex to locate “Materials and Methods” headings, applies heuristics to skip false positives, and saves `methods_subtitles_<PMID>.csv`.

**Workflow Overview**
Those parsers are used in the main file [./07_full_text_retrieval/extract_methods_from_cadmus.py](./07_full_text_retrieval/extract_methods_from_cadmus.py). The workflow is as follows:
1. **Load metadata**  
 Read `retrieved_df2.json.zip` into a DataFrame of PMIDs and parse paths.
2. **Filter out unwanted rows** (this is relevant only for the MS studies for which all PMIDs were fetched, but not all were relevant)
    - Exclude PMIDs in “wrong study type” CSVs.  
    - Remove any row where `xml = 0 && html = 0 && pdf = 0 && plain = 0`.
3. **Attempt each parser in order**  
 For each remaining PMID the parsers are attempted in the following order: XML -> HTML -> PDF -> Plain Text. As soon as one returns success, record its subtitle count and move on.
4. **Logging & Summaries**  
    - Each parser writes logs under `materials_methods/logs/<format>/`.  
    - If no parser succeeds, the PMID is appended to `logs/missing_files.txt`.  
    - At the end, per-format summary files (`summary_stats_<format>.txt`) and an overall summary (`overall_summary_stats.txt`) are generated.

### Sentences extraction
The sentences extraction from the full text is done in [./07_full_text_retrieval/extract_sentences.py](./07_full_text_retrieval/extract_sentences.py). It uses the `nltk` library to tokenize the text into sentences. The script processes each methods section file and saves the sentences in a structured format for further analysis.
1. Reads raw text from a specified column in an input CSV.
2. Splits text into “preliminary” sentences using NLTK’s Punkt tokenizer.
3. Re-tokenizes each preliminary sentence into word/punctuation tokens -> needed espiecially for scientific texts.
4. Merges consecutive sentences if they match “no-split” patterns (false positive splits).
5. Writes out a CSV where each row corresponds to one (possibly merged) sentence.
   
## Full-text information extraction

### Regex-based extraction

This script ([./08_IE_full_text/regex_runner.py](./08_IE_full_text/regex_runner.py)) applies one or more regex-based classifiers to a CSV file containing text data. It supports the following classification categories: `sex`, `species`, `welfare`, `blinding`, `randomization`,  `assay` and `age`. You can run a specific classifier or all of them in sequence.

Each classifier processes a specified text column and outputs a CSV file with encoded prediction results. In the current imoplementation, the text column contains the sentences extracted from the methods section of the full text. The regex patterns are defined in a separate file, which is loaded at runtime. They can be found in [./08_IE_full_text/regex_classifiers/](./08_IE_full_text/regex_classifiers/) and are organized by classifier type, and abstracted in the base class [./08_IE_full_text/classifiers/regex_base.py](./08_IE_full_text/classifiers/regex_base.py).

This abstract base class (`RegexClassifier`) defines the structure for all regex-based classifiers used in the project. It ensures a consistent interface and behavior across classifiers.

Each subclass must implement:
- `compile_patterns(self)`: Compiles the regular expressions needed for classification.
- `classify(self, text: str)`: Applies classification logic to the input text and returns a prediction.

Helper methods:
- `_find_first_match(regex_obj, text)`: Returns the first regex match, if any.
- `_find_all_matches(regex_obj, text)`: Returns all matches as a list.

All regex-based classifiers (e.g., `SexClassifier`, `BlindingClassifier`) inherit from this base class.

### Special Case: Assay/Readout Information Extraction

The `AssayClassifier` is a subclass of `RegexClassifier` designed for extracting assay and outcome readout information. It uses a CSV vocabulary of canonical test names and synonyms to detect mentions of experimental assays in text. The final vocabulary is saved in [08_IE_full_text/data/assay_extraction/assay_final_harmonized_with_enriched_synonyms.csv](08_IE_full_text/data/assay_extraction/assay_final_harmonized_with_enriched_synonyms.csv). The processing of the assay vocabulary can be found in [03_IE_tasks/Harmonize_Outcomes_Assay_Library.ipynb](03_IE_tasks/Harmonize_Outcomes_Assay_Library.ipynb).

To build the assay vocabulary:

- Several review papers across domains (e.g., neuroscience, physiology, pharmacology) were reviewed. The outputs are in [03_IE_tasks/data/related/outcomes_endpoints](03_IE_tasks/data/related/outcomes_endpoints).
- For each article, a structured assay table was constructed with the help of ChatGPT in the following format:

```
Test Name	Measure	Outcome Domain	Subdomain
Microarray analysis	Fold-change in expression or hybridization intensity across all array probes	Molecular & Cellular	Transcriptomics
SNP analysis	Allele frequency and genotype calls across genome-wide SNP panels	Molecular & Cellular	Genomics
```

- Each test was assigned to one of five outcome domains:

  1. **Behavioural**  
     *Definition:* Any overt animal action or choice—motor, sensorimotor, cognitive, social, or affective/pain-related  
     *(e.g., rotarod, open field, elevated plus maze, hyperreflexia, seizure)*

  2. **Imaging**  
     *Definition:* Includes **in vivo** medical imaging (e.g., MRI, PET, CT). Does **not** include microscopy or photographic images.

  3. **Histology**  
     *Definition:* Structural and cellular-level visualization methods, typically involving tissue sectioning and staining (e.g., Nissl, H&E, LFB), often coupled with various types of microscopy.

  4. **Physiology**  
     *Definition:* Vital signs, metabolic readouts, body composition, pharmacokinetics, and survival  
     *(e.g., heart rate, blood pressure, body weight, blood flow, EEG, EMG)*

  5. **Molecular & Cellular**  
     *Definition:* Covers analyses that quantify or characterize molecular components (DNA, RNA, proteins, metabolites) or cellular properties (e.g., cell types, activation states) using laboratory assays  
     *(e.g., Western blot, qPCR, RNA-seq, ELISA, mass spectrometry, FACS)*

- Test names were harmonized into canonical forms. 
- Synonyms for each test were generated and enriched using ChatGPT. The synonyms are in the file [03_IE_tasks/data/related/outcomes_endpoints/harmonized_synonyms_GPT.xlsx](03_IE_tasks/data/related/outcomes_endpoints/harmonized_synonyms_GPT.xlsx). 

The final harmonized vocabulary was saved as a CSV file and used to compile regex patterns for each assay category. These patterns are used at runtime to extract structured assay information from unstructured text.

The classifier returns:
- A multi-hot vector indicating matched assay categories
- A list of matched outcome domains
- A mapping from matched domains to the canonical test names found


#### Special Case: Age Information Extraction 

Age classification follows a multi-step pipeline due to the high likelihood of false positives from simple regex matching. The process includes regex filtering, machine learning classification, and LLM-based age value extraction. The scripts for this are located in the [./08_IE_full_text/age_classifier/](./08_IE_full_text/age_classifier/) directory.


**1. Sentence Filtering with Regex**

After applying the regex-based age classifier, sentences containing age-related keywords are filtered using:

`./08_IE_full_text/extract_age_sentences.py`  
→ Extracts only sentences matched by age-related regex patterns.


**2. Sentence Classification with Machine Learning**

Due to many false positives, a BERT-based classifier is trained to distinguish between:
- **True Age Sentences** (actual age info)
- **False Positives** (irrelevant mentions)

Training:
`./08_IE_full_text/age_classifier/Age_Sentence_Classifier_Train.ipynb`  
→ Trained on manually annotated age-labeled data.

Inference:
`./08_IE_full_text/age_classifier/age_sent_clssifier_bert.py`  
Run via: `./08_IE_full_text/age_classifier/run_age_sent_classifier_bert.sh`


**3. Age Value Extraction with LLM**

To extract the actual age values from true age sentences, a lightweight fine-tuned LLM is used.

Model:
`./08_IE_full_text/age_classifier/LLM_Unsloth.ipynb`  
→ A fine-tuned version of Unsloth (optimized for fast inference).

Inference:
`./08_IE_full_text/age_classifier/LLM_Unsloth_Inference.ipynb`  
→ Applies the LLM to extract age values from each true age sentence.

**Postprocessing**

- [./08_IE_full_text/clean_age_llm_predictions.py](./08_IE_full_text/clean_age_llm_predictions.py)  
  Cleans and processes the LLM predictions, including abbreviation expansion and unique entity extraction.

### NER-based extraction
See above NER section for details on the NER-based extraction of conditions and interventions from the full text.



**Postprocessing**
- [./08_IE_full_text/clean_ner_predictions.py](./08_IE_full_text/clean_ner_predictions.py)  
  Cleans and processes the NER predictions, including abbreviation expansion and unique entity extraction.
- [./08_IE_full_text/convert_animal_nr_to_numeric.py](./08_IE_full_text/convert_animal_nr_to_numeric.py)  
  Converts animal number predictions from text to numeric format.

#### Animal Number Extraction
After the NER-based extraction, we ensure that the number extracted likely refers to the number of animals used in the study. For that we find the mention of the number extracted in the methods section, and check if it is in the context of "animals" or a specific species. This is done in the script [08_IE_full_text/clean_animal_nr.py](08_IE_full_text/clean_animal_nr.py). Only the NER outputs matching this criteria are kept for further processing.

In a second step, the valid animal number predictions are processed to convert them from text to numeric format. This is done in the script [08_IE_full_text/convert_animal_nr_to_numeric.py](08_IE_full_text/convert_animal_nr_to_numeric.py). 

### Document-Level Aggregation Script

The script [./08_IE_full_text/map_sent_to_doc_level.py](./08_IE_full_text/map_sent_to_doc_level.py) processes sentence-level regex predictions and converts them into document-level outputs. It includes logic for standard categories and special handling for species predictions.

**Main Workflow**
- Reads sentence-level CSV prediction files (e.g., `sex_predictions.csv`, `blinding_predictions.csv`).
- Applies `document_level_strict_zero_fallback()` to generate document-level results.
- Processes `species_predictions.csv` separately using `process_species_exclude_singletons()` for improved logic.
- Saves all outputs as new CSVs with `_doc_level_predictions.csv` suffix.

Each output includes:
- `PMID`
- Final label (`prediction_encoded_label` or `species`)
- Supporting sentence IDs

## Joining preclinical and clincal data

