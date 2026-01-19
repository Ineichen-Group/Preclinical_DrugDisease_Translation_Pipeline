import re
from typing import Tuple

from .regex_base import RegexClassifier

class SampleSizeCalcClassifier(RegexClassifier):
    """
    Detects whether a text reports a sample size calculation / power analysis.
    Returns:
      (1, "sample-size-present")          if positive cues are found,
      (2, "sample-size-not-performed")    if explicit negatives are found,
      (0, "sample-size-not-reported")     otherwise.
    """

    def compile_patterns(self) -> None:
        # -------- Negative cues (explicitly NOT performed) --------
        negative_parts = [
            r"\bno\s+(?:a\s+priori\s+)?(?:sample[-\s]?size\s+(?:calculation|estimation|determination)|power\s+analysis)\b",
            r"\b(?:sample[-\s]?size|power\s+analysis)\s+(?:was|were)\s+not\s+(?:performed|conducted|done|carried\s+out|calculated|estimated|determined)\b",
            r"\bno\s+sample[-\s]?size\s+(?:was|were)\s+(?:calculated|estimated|determined|computed)\b",
            r"\b(?:without|lacking)\s+(?:a\s+priori\s+)?(?:sample[-\s]?size\s+(?:calculation|estimation|determination)|power\s+analysis)\b",
            r"\bno\s+(?:power\s+analysis|power\s+calculation)\b",
        ]
        self._neg_rx = re.compile("|".join(negative_parts), re.IGNORECASE)

        # -------- Positive cues (performed / justified) --------
        positive_parts = [
            # direct mentions of calculation/estimation/determination
            r"\bsample[-\s]?size(?:s)?\s+(?:was|were)\s+(?:calculated|estimated|determined|computed|predetermined)\b",
            r"\b(?:we|authors?)\s+(?:calculated|estimated|determined)\s+the\s+sample[-\s]?size\b",
            r"\bsample[-\s]?size\s+(?:calculation|estimation|determination)\s+(?:was|were)\s+(?:performed|conducted|done|carried\s+out)\b",

            # power analysis explicitly mentioned
            r"\b(?:power\s+analysis|power\s+calculation)\s+(?:was|were)\s+(?:performed|conducted|done|carried\s+out)\b",
            r"\b(?:power\s+analysis|power\s+calculation)\b",

            # “determined by/based on …” formulations
            r"\b(?:samples?\s+sizes?|number\s+of\s+(?:mice|rats|animals|subjects)(?:\s+per\s+group)?)\s+was\s+(?:determined|set|chosen)\s+(?:by|based\s+on|according\s+to)\b",

            # power + alpha style statements implying calc
            r"\bpower\s*(?:=|of)\s*\d+(?:\.\d+)?\s*(?:%|percent)?\b.*?\b(?:alpha|α)\s*(?:=|of)?\s*0?\.\d+\b",
            r"\b(?:alpha|α)\s*(?:=|of)?\s*0?\.\d+\b.*?\bpower\s*(?:=|of)\s*\d+(?:\.\d+)?\s*(?:%|percent)?\b",

            # “ensure adequate power / sufficient to attain statistical power”
            r"\bensure\s+adequate\s+power\b",
            r"\bsufficient\s+to\s+attain\s+statistical\s+power\b",
        ]
        self._pos_rx = re.compile("|".join(positive_parts), re.IGNORECASE)

    def classify(self, text: str) -> Tuple[int, str]:
        # Explicit negative beats everything
        if self._neg_rx.search(text):
            return 2, "sample-size-not-performed"

        # Any positive signal → present
        if self._pos_rx.search(text):
            return 1, "sample-size-present"

        # Otherwise unknown
        return 0, "sample-size-not-reported"
