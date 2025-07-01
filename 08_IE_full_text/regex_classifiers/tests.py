# test_classifier.py

from assay_classifier import AssayClassifier

# Load your classifier from the CSV
classifier = AssayClassifier("08_IE_full_text/data/assay_extraction/assay_final_harmonized_with_enriched_synonyms.csv")

# Example input
text = """
A s.c. U87MG human glioblastoma xenograft model was used to determine maximum tolerated dose (MTD), biodistribution, dose response, and efficacy of 90Y-Abegrin. 
Antitumor efficacy was also characterized in an orthotopic U87MG and in a HT-29 colorectal cancer model, a low integrin-expressing carcinoma. 
Small-animal positron emission tomography imaging was used to correlate histologic findings of treatment efficacy.

MTD and dose response analysis revealed 200 microCi per mouse as appropriate treatment dose with hepatic clearance and no organ toxicity. 
90Y-Abegrin-treated U87MG tumor mice showed partial regression of tumor volume, with increased tumor volumes in 90Y-IgG, Abegrin, and saline groups. 
18F-FDG imaging revealed a reduction of cell proliferation and metabolic activity whereas 18F-FLT reflected decreased DNA synthesis in the 90Y-Abegrin group.
Ki67 analysis showed reduced proliferative index and quantitative terminal deoxynucleotidyl transferase dUTP nick-end labeling-positive analysis revealed increased DNA fragmentation and apoptosis in 90Y-Abegrin animals. 
CD31 and 4',6-diamidino-2-phenylindole staining showed increased vascular fragmentation and dysmorphic vessel structure in 90Y-Abegrin animals only. 
Orthotopic U87MG tumors treated with 90Y-Abegrin displayed reduced tumor volume. HT-29 tumors showed no significant difference among the various groups.
"""

# Classify the text
vector, labels, matches = classifier.classify(text)

# Print results
print("Multi-hot Vector:", vector)
print("Matched Labels:", labels)
print("Matches by Label:")
for label, canonical_list in matches.items():
    print(f"  {label}: {canonical_list}")
