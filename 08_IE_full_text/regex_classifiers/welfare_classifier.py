# classifiers/welfare_classifier.py

import re
from typing import Tuple

from .regex_base import RegexClassifier

class WelfareClassifier(RegexClassifier):
    """
    Detects whether a given text contains any “animal welfare” keywords,
    excluding common false‐positive contexts. Returns:
      (1, "welfare-present")  if at least one valid welfare keyword is found,
      (0, "welfare-not-reported") otherwise.
    """

    def compile_patterns(self) -> None:
        """
        Compile the “welfare” and “exclude” regexes once at instantiation.
        """
        # The main “welfare keyword” pattern (raw string)
        welfare_keywords_pattern = (
            r"\b(approved|approval|authorized|reviewed|"
            r"carried out (in|under)? (strict )?(accordance|compliance|adherence)? (with|to)|"
            r"conducted (in|under)? (strict )?(accordance|compliance|adherence)? (with|to)|"
            r"performed (in|under)? (strict )?(accordance|compliance|adherence)? (with|to)|"
            r"in (strict )?(accordance|compliance|adherence)? (with|to)|"
            r"according to( the)? (institutional |national |international |ethical )?(guidelines|policy|regulations|standards|rules|principles)|"
            r"conformed (to|with)|"
            r"complied (with|to)?( relevant)? (guidelines|regulations|standards)?|"
            r"treated according to( the)? (guidelines|regulations|standards)?|"
            r"in (compliance|accord|agreement|adherence) (with|to)|"
            r"(Institutional|Animal|Ethics) Committee|"
            r"(IACUC|CPCSEA)|"
            r"Guide for the Care and Use of Laboratory Animals|"
            r"Directive (86/609|2010/63)[/\w]*|"
            r"ARRIVE guidelines|"
            r"Animal Care|"
            r"ethical treatment|"
            r"Declaration of Helsinki|"
            r"ethical (standards|guidelines|policies)|"
            r"international (laws|standards|guidelines)|"
            r"regulations (of|from|by) .*animal.*"
            r")\b"
        )

        # Terms that, if present, invalidate a “welfare keyword” match (false positives)
        exclude_pattern = (
            r"\b(human|humans|author[s]?|subject[s]?|patient[s]?|volunteer[s]?|donor[s]?|"
            r"method described by|manufacturer(?:'s)?|company|euthanized|inc\b|ltd\b|"
            r"sigma|fisher|thermo|roche|abbott|bio-rad)\b"
        )

        # Compile both patterns once, with IGNORECASE
        self._welfare_regex = re.compile(welfare_keywords_pattern, flags=re.IGNORECASE)
        self._exclude_regex = re.compile(exclude_pattern, flags=re.IGNORECASE)

    def classify(self, text: str) -> Tuple[int, str]:
        """
        Return (1, 'welfare-present') if:
          - self._welfare_regex finds at least one match in `text`, AND
          - self._exclude_regex does NOT find any matches in `text`.
        Otherwise return (0, 'welfare-not-reported').
        """
        # 1) If no welfare keyword at all, immediately “not‐reported”
        if not self._welfare_regex.search(text):
            return 0, "welfare-not-reported"

        # 2) If any exclude term is present, override to “not reported”
        if self._exclude_regex.search(text):
            return 0, "welfare-not-reported"

        # 3) Otherwise, we have a valid welfare mention
        return 1, "welfare-present"
