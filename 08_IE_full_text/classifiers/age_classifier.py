# classifiers/age_classifier.py

import re
from typing import Tuple

class AgeClassifier:
    """
    Detects whether a given text contains any “age” keywords or age‐related expressions.
    Returns:
      (1, "age-present")         if at least one age pattern matches,
      (0, "age-not-reported")    otherwise.
    """

    def __init__(self):
        # Compile the age‐keyword regex once upon instantiation
        self.compile_patterns()

    def compile_patterns(self) -> None:
        """
        Compile the single age_keywords_pattern into a re.Pattern with IGNORECASE.
        """
        age_keywords_pattern = (
            r'\b('
            r'age|ages|aged|aging|old|older|young|adult|adults|mature|senescent|'
            r'neonatal|neonate|newborn|pup|pups|juvenile|weanling|weaning|'
            r'postnatal|prenatal|prepubescent|fetal|fetus|fetuses|'
            r'day[-\s]?old|week[-\s]?old|month[-\s]?old|year[-\s]?old|'     # hyphen or space
            r'dayold|weekold|monthold|yearold|'                            # concatenated forms
            r'\d+\s*[-–to]+\s*\d+\s*(day|week|wk|month|year|yr)s?|'           # ranges like 6-8 weeks
            r'\d+\s*(day|week|wk|month|year|yr)s?(?!\s*old)|'                 # standalone like "8 weeks"
            r'after birth|post[-\s]?birth'
            r')\b'
        )
        self._age_regex = re.compile(age_keywords_pattern, flags=re.IGNORECASE)

    def classify(self, text: str) -> Tuple[int, str]:
        """
        Return (1, 'age-present') if self._age_regex finds a match in `text`,
        otherwise (0, 'age-not-reported').
        """
        if self._age_regex.search(text):
            return 1, "age-present"
        else:
            return 0, "age-not-reported"
