# classifiers/base.py

import re
from abc import ABC, abstractmethod
from typing import Any, Optional, List

class RegexClassifier(ABC):
    """
    Abstract base for any regex-driven classifier.
    Subclasses must implement:
      - compile_patterns(self) → None
      - classify(self, text: str) → Any
    """

    def __init__(self):
        # Compile patterns once at instantiation
        self.compile_patterns()

    @abstractmethod
    def compile_patterns(self) -> None:
        """
        Subclasses should compile any regex(es) they need and store them
        in instance variables, e.g.:
            self._female_regex = re.compile(r"...", flags=re.IGNORECASE)
        """
        pass

    @abstractmethod
    def classify(self, text: str) -> Any:
        """
        Run this classifier’s logic on the given text.
        Return whatever data structure is appropriate (e.g. a tuple).
        """
        pass

    def _find_first_match(self, regex_obj: re.Pattern, text: str) -> Optional[re.Match]:
        return regex_obj.search(text)

    def _find_all_matches(self, regex_obj: re.Pattern, text: str) -> List[re.Match]:
        return list(regex_obj.finditer(text))
