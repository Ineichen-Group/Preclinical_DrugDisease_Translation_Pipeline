# classifiers/blinding_classifier.py

import re
from typing import Tuple

from .regex_base import RegexClassifier

class BlindingClassifier(RegexClassifier):
    """
    Detects whether a given text contains any “blinding” keywords (e.g. “blind”, “were blind”).
    Returns:
      (1, "blinding-present")   if at least one keyword is found,
      (0, "blinding-not-reported") otherwise.
    """

    def compile_patterns(self) -> None:
        """
        Compile the blinding regex once at instantiation.
        """
        # Very simple pattern—looks for “blind” or “were blind” (case‐insensitive)
        blinding_keywords_pattern = r"blinded|were blind|blind"

        # Compile with IGNORECASE so “Blind” or “BLIND” also match
        self._blinding_regex = re.compile(blinding_keywords_pattern, flags=re.IGNORECASE)

    def classify(self, text: str) -> Tuple[int, str]:
        """
        Return (1, 'blinding-present') if self._blinding_regex finds a match,
        otherwise (0, 'blinding-not-reported').
        """
        if self._blinding_regex.search(text):
            return 1, "blinding-present"
        else:
            return 0, "blinding-not-reported"
