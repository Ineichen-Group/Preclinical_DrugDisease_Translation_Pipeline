# classifiers/sex_classifier.py

import re
from typing import Tuple, Dict, List, Optional

class SexClassifier:
    """
    Scans text for “sex-female”, “sex-male”, or “sex-both” patterns.
    Returns a tuple: (numeric_code, label_str).
    """

    def __init__(self, window: int = 12):
        self.window = window  # number of tokens on each side for false-context check
        self.compile_patterns()

    def compile_patterns(self) -> None:
        """
        Build self._patterns as a dict: label → [re.Pattern, …]
        and define self._label_to_code mapping.
        """
        # -------- FALSE CONTEXT TERMS (hardcoded, multiline) --------
        FALSE_CONTEXT_TERMS: str = r"""(?ix)
        \b(
            patients?              # 'patient'/'patients'
          | females?\s+versus\s+males?   # 'females versus males'
        )\b
        """
        self._false_context = re.compile(FALSE_CONTEXT_TERMS, re.IGNORECASE | re.VERBOSE)


        raw_patterns: Dict[str, List[str]] = {
            "sex-female": [
                r"\bfemales?\b",
                r"\bmaternal\b",
                r"\bmothers?\b",
                r"\bsisters?\b",
                r"\bpregnant?\b",
            ],
            "sex-male": [
                r"\bmales?\b",
                r"\bfathers?\b",
                r"\bbrothers?\b",
            ],
            "sex-both": [
                r"\beither sex(?:es)?\b",
                r"\bboth sex(?:es)?\b",
                r"\b(?:of|the)\s+either sex(?:es)?\b",
            ],
        }

        self._patterns: Dict[str, List[re.Pattern]] = {
            label: [re.compile(p, re.IGNORECASE) for p in patt_list]
            for label, patt_list in raw_patterns.items()
        }

        self._label_to_code: Dict[str, int] = {
            "sex-both": 3,
            "sex-female": 1,
            "sex-male": 2,
            "sex-not-reported": 0,
        }

    def _find_first_match(self, regex_obj: re.Pattern, text: str, context: int = 50) -> Optional[re.Match]:

        match = regex_obj.search(text)
        if not match:
            return None

        start, end = match.start(), match.end()

        # show context snippet
        snippet_start = max(0, start - context)
        snippet_end = min(len(text), end + context)
        snippet = text[snippet_start:snippet_end]
        #print(f"Found match: {match.group(0)!r}")
        #print(f"Context: ...{snippet}...")

        # ---- false-context window check ----
        tokens = list(re.finditer(r"\b\w+\b", text))
        overlapping_indices = [
            i for i, tok in enumerate(tokens)
            if (tok.start() < end and tok.end() > start)
        ]
        if overlapping_indices:
            for idx in overlapping_indices:
                start_idx = max(0, idx - self.window)
                end_idx = min(len(tokens), idx + self.window + 1)

                char_start = tokens[start_idx].start()
                char_end = tokens[end_idx - 1].end()
                window_snippet = text[char_start:char_end]
                
                if self._false_context.search(window_snippet):
                    #print("→ Skipped: FALSE_CONTEXT_TERMS found within token window.")
                    return None

        return match

    def classify(self, text: str) -> Tuple[int, str]:
        found: Dict[str, bool] = {label: False for label in self._patterns}

        for label, compiled_list in self._patterns.items():
            for regex_obj in compiled_list:
                if self._find_first_match(regex_obj, text):
                    found[label] = True
                    break

        if found["sex-both"] or (found["sex-female"] and found["sex-male"]):
            final_label = "sex-both"
        elif found["sex-female"]:
            final_label = "sex-female"
        elif found["sex-male"]:
            final_label = "sex-male"
        else:
            final_label = "sex-not-reported"


        numeric_code = self._label_to_code[final_label]
        return numeric_code, final_label
