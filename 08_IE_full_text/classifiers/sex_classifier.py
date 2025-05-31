# classifiers/sex_classifier.py

import re
from typing import Tuple, Dict, List

class SexClassifier:
    """
    Scans text for “sex-female”, “sex-male”, or “sex-both” patterns.
    Returns a tuple: (numeric_code, label_str).
    """

    def __init__(self):
        # Compile all patterns once at instantiation
        self.compile_patterns()

    def compile_patterns(self) -> None:
        """
        Build self._patterns as a dict: label → [re.Pattern, …]
        and define self._label_to_code mapping.
        """
        # Raw regex patterns for each label, hard‐coded here:
        raw_patterns: Dict[str, List[str]] = {
            "sex-female": [
                r"\bfemales?\b",
                r"\bmaternal\b",
                r"\bmothers?\b",
                r"\bsisters?\b",
            ],
            "sex-male": [
                r"\bmales?\b",
                r"\bfathers?\b",
                r"\bbrothers?\b",
            ],
            "sex-both": [
                r"\beither sex(?:es)\b",
                r"\both sex(?:es)\b",
                r"\b(?:of|the)\seither sex\b",
            ],
        }

        # Compile each raw pattern into a re.Pattern (IGNORECASE)
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for label, patt_list in raw_patterns.items():
            self._patterns[label] = [
                re.compile(p, flags=re.IGNORECASE) for p in patt_list
            ]

        # Define a mapping from label strings to numeric codes
        self._label_to_code: Dict[str, int] = {
            "sex-both": 0,
            "sex-female": 1,
            "sex-male": 2,
            "sex-not-reported": 3,
        }

    def _find_first_match(self, regex_obj: re.Pattern, text: str) -> re.Match:
        """
        Convenience method: return the first match object or None.
        """
        return regex_obj.search(text)

    def classify(self, text: str) -> Tuple[int, str]:
        """
        Check each label’s compiled patterns against `text`.
        If both female & male matched → 'sex-both'
        If only female → 'sex-female'
        If only male → 'sex-male'
        Else → 'sex-not-reported'
        Then look up the numeric code in self._label_to_code and return (code, label).
        """
        found: Dict[str, bool] = {label: False for label in self._patterns}

        for label, compiled_list in self._patterns.items():
            for regex_obj in compiled_list:
                if self._find_first_match(regex_obj, text):
                    found[label] = True
                    break

        if found.get("sex-female", False) and found.get("sex-male", False):
            final_label = "sex-both"
        elif found.get("sex-female", False):
            final_label = "sex-female"
        elif found.get("sex-male", False):
            final_label = "sex-male"
        else:
            final_label = "sex-not-reported"

        numeric_code = self._label_to_code[final_label]
        return numeric_code, final_label
