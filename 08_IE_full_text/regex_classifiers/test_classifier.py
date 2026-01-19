# test_classifier.py

from .assay_classifier import AssayClassifier
from .species_classifier import SpeciesClassifier
from .sex_classifier import SexClassifier
from .welfare_classifier import WelfareClassifier
from pathlib import Path
# run with python -m regex_classifiers.test_classifier from 08_IE_full_text

BASE_DIR = Path(__file__).resolve().parents[1]  # 08_IE_full_text/
DATA_PATH = BASE_DIR / "data/assay_extraction/assay_final_harmonized_with_enriched_synonyms.csv"

# Load your classifier from the CSV
classifier = AssayClassifier(DATA_PATH)
classifier_species = SpeciesClassifier()
sex_classifier = SexClassifier()
welfare_classifier = WelfareClassifier()

# Example input
text = """
ethical treatment of animals was carried out 
Amrut rat and mice feed
7-9 months-old animals of both sexes were
Laboratory Rabbit Diet
The phosphorylated form was normalized against β-tubulin (Cell Signaling Cat. #2146S). 
Another Wes was run for the total amount of IR (Cell Signaling Cat. #3025S). 
guinea-pigs (Cavia porcellus) were purchased from Charles River Laboratories (Wilmington, MA, USA) and housed in the animal facility of the University of Barcelona."""

# Classify the text
vector, labels, matches = classifier.classify(text)
vector_species, labels_species = classifier_species.classify(text)
numeric_code, sex_label = sex_classifier.classify(text)
welfare_code, welfare_label = welfare_classifier.classify(text)

# Print results
print("Multi-hot Vector:", vector)
print("Matched Labels:", labels)
print("Matches by Label:")
for label, canonical_list in matches.items():
    print(f"  {label}: {canonical_list}")
    
print("\nSpecies Classifier Multi-hot Vector:", vector_species)
print("Species Matched Labels:", labels_species)
for label in labels_species:
    print(f"  {label} patterns matched.")
    
print("\nSex Classifier Results:")
print("Numeric Code:", numeric_code)
print("Label:", sex_label)

print("\nWelfare Classifier Results:")
print("Numeric Code:", welfare_code)
print("Label:", welfare_label)