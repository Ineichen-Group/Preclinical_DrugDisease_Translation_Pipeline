import re
import csv
from typing import Dict, List, Tuple
from collections import defaultdict
from regex_base import RegexClassifier

class AssayClassifier(RegexClassifier):
    """Classifier to detect assays by category using CSV-driven canonical/synonym mapping."""

    def __init__(self, csv_path: str = None):
        if csv_path is None:
            # Set default relative or absolute path here
            default_path = "./data/assay_extraction/assay_final_harmonized_with_enriched_synonyms.csv"
            csv_path = default_path
        self.csv_path = csv_path
        self.ASSAY_LABELS: List[str] = []  # Will be inferred from data
        self._label_to_index: Dict[str, int] = {}
        self._patterns: Dict[str, re.Pattern] = {}
        self._synonym_to_canonical: Dict[str, Dict[str, str]] = {}
        super().__init__()

    def compile_patterns(self) -> None:
        """Parse CSV and compile regex patterns and synonym→canonical mappings by domain."""

        domain_synonyms = defaultdict(list)  # Outcome Domain → list of synonyms
        domain_synonym_to_canonical = defaultdict(dict)  # Outcome Domain → {synonym → canonical}

        with open(self.csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                
                canonical = row["Canonical Name"].strip()
                domain = row["Outcome Domain"].strip()
                synonyms = [s.strip().lower() for s in row["Synonym"].split(";") if s.strip()]

                for syn in synonyms:
                    domain_synonym_to_canonical[domain][syn] = canonical
                    domain_synonyms[domain].append(syn)

        self.ASSAY_LABELS = sorted(domain_synonyms.keys())
        self._label_to_index = {label: idx for idx, label in enumerate(self.ASSAY_LABELS)}
        self._synonym_to_canonical = dict(domain_synonym_to_canonical)

        # Compile regex patterns per domain
        for domain, synonyms in domain_synonyms.items():
            escaped = sorted([re.escape(s) for s in set(synonyms)], key=len, reverse=True)
            pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
            self._patterns[domain] = re.compile(pattern)

    def classify(self, text: str) -> Tuple[List[int], List[str], Dict[str, List[str]]]:
        """
        Classify the text into assays.
        
        Returns:
            - vector: multi-hot vector of matched categories.
            - found_labels: list of category names matched.
            - matches_by_label: {category → list of matched canonical names}
        """
        vector = [0] * len(self.ASSAY_LABELS)
        found_labels: List[str] = []
        matches_by_label: Dict[str, List[str]] = {}

        for label in self.ASSAY_LABELS:
            pattern = self._patterns[label]
            matches = self._find_all_matches(pattern, text)
            if matches:
                vector[self._label_to_index[label]] = 1
                found_labels.append(label)
            
                canonicals = [
                   self._synonym_to_canonical[label][m.group(0).lower()]
                    for m in set(matches)
                ]
                canonicals = list(dict.fromkeys(canonicals)) # Remove duplicates while preserving order
                matches_text_pos = [
                    f"{m.group(0)} ({m.start()}-{m.end()})"
                    for m in matches
                ]
                matches_by_label[label] =  canonicals

        return vector, found_labels, matches_by_label
