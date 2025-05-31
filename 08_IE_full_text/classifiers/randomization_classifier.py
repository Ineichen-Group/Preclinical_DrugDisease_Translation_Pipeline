# classifiers/randomization_classifier.py

import re
from typing import Tuple

from .regex_base import RegexClassifier

class RandomizationClassifier(RegexClassifier):
    """
    Detects whether a given text contains any “randomization” keywords (e.g. “randomized”, “randomly assigned”).
    Returns:
      (1, "randomization-present")   if at least one pattern matches,
      (0, "randomization-not-reported") otherwise.
    """

    def compile_patterns(self) -> None:
        """
        Compile the randomization regex once at instantiation.
        """
        randomization_keywords_pattern = (
            r"\b("
            r"randomized(?: control(?:led)? (?:trial|study))?|"
            r"randomized (?:mice|rats|animals|groups)|"
            r"(?:were|was|had been)?\s*randomly (?:assigned|divided|grouped|grouping|selected|allocated|placed|chosen|culled|distributed)|"
            r"(?:assigned|divided|grouped|grouping|allocated|chosen|selected|culled) randomly|"
            r"(?:assigned|divided|grouped|allocated|placed|selected|chosen) at random|"
            r"pseudorandomly (?:assigned|placed|culled|selected)?|"
            r"random assignment|"
            r"random allocation|"
            r"randomization (?:was performed|of animals)?|"
            r"randomly allocated|"
            r"animals were randomized to|"
            r"randomized into (?:groups|treatment groups)|"
            r"randomly selected (?:mice|rats|subjects|animals)"
            r")\b"
        )

        # Compile with IGNORECASE
        self._randomization_regex = re.compile(randomization_keywords_pattern, flags=re.IGNORECASE)

    def classify(self, text: str) -> Tuple[int, str]:
        """
        Return (1, 'randomization-present') if self._randomization_regex finds a match,
        otherwise (0, 'randomization-not-reported').
        """
        if self._randomization_regex.search(text):
            return 1, "randomization-present"
        else:
            return 0, "randomization-not-reported"
