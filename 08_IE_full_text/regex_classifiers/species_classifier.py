# classifiers/species_classifier.py

import regex as re
from typing import List, Tuple, Optional, Match


class SpeciesClassifier:
    """
    Scans text for any of the species (mouse, rat, rabbit, monkey, guinea pig, dog, cat, pig, species-other),
    skipping matches that fall in a “false context” (e.g. immunology terms). 
    Returns: ([binary_vector_over_species_labels], [list_of_matched_labels]).
    """

    # Class‐level list of species labels in fixed order
    SPECIES_LABELS: List[str] = [
        "mouse",
        "rat",
        "rabbit",
        "monkey",
        "guinea pig",
        "species-other",
        "dog",
        "cat",
        "pig",
    ]

    # Raw regex patterns for each species label
    RAW_PATTERNS: dict[str, List[str]] = {
        "rat": [
            r"\brats?\b"
            # not feed / diet product names (e.g., "rat and mice feed")
            r"(?!\s+(?:and\s+)?mice\s+feed\b)"
            r"(?!\s+feed\b)"
            r"(?!\s+diet\b)",
        ],
        "mouse": [
            r"\bmouse\b"
            r"(?!\s+MOG\b)"                 # not "mouse MOG"
            r"(?!\s+(?:feed|diet)\b)",      # not "mouse feed/diet"

            r"\bmice\b"
            r"(?!\s+(?:feed|diet)\b)",      # not "mice feed/diet"
        ],
        "cat": [
            # Match "cat" / "cats" as the experimental animal ONLY.
            # Explicitly EXCLUDES:
            # 1) Predator-odor / object contexts:
            #    - cat litter
            #    - cat odor / odour (including odor-exuding, odor cues)
            #    - cat collar, cat cue
            #    - worn by a (domestic) cat
            #
            # 2) Reagent / catalog references:
            #    - Cat., Cat#, Cat No./Nr./Num./Number
            #    - "Cat: 1409-rPEP-02"
            #    - Vendor phrases (e.g., Cell Signaling Cat. #)
            #
            # 3) Author name contexts:
            #    - "Cats J."
            #
            # 4) Molecular biology / enzyme contexts:
            #    - CAT enzyme, CAT activity, CAT assay
            #    - CAT gene, reporter, LTR-CAT constructs
            #    - CAT levels in cell extracts
            #
            # Rationale:
            # - "cat" is highly ambiguous in biomedical text.
            # - This rule prioritizes avoiding false positives over capturing every mention.
            #
            # Match "cat" / "cats" as the experimental animal ONLY.
            r"(?<![A-Z]\.\s)"                # not initials like "A. Cat"
            r"(?<!concentration of\s)"       # not "concentration of CAT ..."
            r"(?<!concentrations of\s)"
            r"(?<!level of\s)"
            r"(?<!levels of\s)"
            r"(?<!\bworn\s+by\s+(?:a|an|the)?\s*(?:domestic\s+)?)"  # NOT "worn by a cat"
            r"\bcat(?:s)?\b"

            # predator-odor / object contexts
            r"(?!\s*(?:domestic\s+)?(?:litter|cue|collar|odou?r(?:[-\s]\w+)*)\b)"

            # catalog / reagent refs
            r"(?!\s*\.\s*#)"
            r"(?!\s*\.\s*[A-Z0-9-]{3,})"
            r"(?!\s*\.?\s*(?:#|no\.?|nr\.?|num\.?|number|id)\b)"
            r"(?!\s*[:#]\s*[\w-]+)"
            r"(?!\s*[A-Z]\.)"
            r"(?!\s*,\s*EC\b)"

            # enzyme / molecular biology contexts
            r"(?![\s\-\(\[]*(?:activity|assay|enzyme|protein|levels?|concentration(?:s)?|expression)\b)"
            r"(?![\s\-\(\[]*(?:gene|reporter|ltr|construct|vector|plasmid|transfection|promoter|mrna|cdna)\b)"

            # cell extracts not necessarily immediately after (window of next ~6 words)
            r"(?!\b(?:\W+\w+){0,6}\W+(?:cell|cells|extracts?)\b)"

        ],
        "dog": [r"\bdogs?\b"],
        "guinea pig": [r"\bguinea[-\s]?pigs?\b"],
        "monkey": [
            r"\bmonkeys?\b",
            # Great Apes & Hominids
            r"\bchimpanzees?\b",
            r"\borangutans?\b",
            r"\bbonobos?\b",
            r"\bgibbons?\b",
            r"\bgorillas?\b",
            # Old World Monkeys (Common in research)
            r"\bmacaques?\b",
            r"\brhesus\b",
            r"\bcynomolgus\b",
            r"\bbaboons?\b",
            r"\bpapio\b",  # Genus for baboons
            r"\bvervets?\b",
            r"\bgrivets?\b",
            r"\bafrican green monkeys?\b",
            # New World Monkeys (Common in research)
            r"\bmarmosets?\b",
            r"\btamarins?\b",
            r"\bsquirrel monkeys?\b",
            r"\bsaimiri\b", # Genus for squirrel monkeys
            r"\bcapuchins?\b",
            # General Research Terms
            r"\bnon-?human primates?\b",
            r"\bnhp\b",
        ],
        "pig": [r"\bpigs?\b", r"\bswines?\b", r"\bpiglets?\b"],
        "rabbit": [
            r"\brabbits?\b"
            r"(?!\s+Ig[GMDAE]?\b)"                           # not rabbit IgG/IgM/IgA…
            r"(?!\s+(?:standard|laboratory|commercial)\s+rabbit\s+diet\b)"  # not Laboratory Rabbit Diet
            r"(?!\s+diet\b)",                                # not rabbit diet (generic)
        ],

        "species-other": [],  # fallback
    }

    # False context terms that invalidate a species match if found in proximity
    FALSE_CONTEXT_TERMS: str = r"""
        (?ix)
        \b(
            antibody|antibodies|
            antiserum|antisera|antigens?|tissues?|dilution|
            monoclonal|polyclonal|wako|Wako|peptides?|
            Ig\s*[A-Z]{1,2}|Ig-?coated|
            mAb|pAb|HRP|APC|FITC|PE|MBP|Cy\d+|myelin\s+basic\s+protein|
            ELISA|immunoblot|western\s+blot|immunostaining|
            conjugated|biotinylated|fluorescent-?labeled|
            GAPDH|tubulin|β-actin|emulsified|emulsion|
            emulsified\s+in|injections?\s+of|activated\s+by|primary\s+cultures?|
            luciferase|peroxidase|polymerase|qPCR|RT-PCR|Taq|
            serum|lysate|recombinant|TG2|anti|OX\d+|CD\d+
        )\b
        |
        # ---- Antibody reagent cues (no vendor handling) ----
    
        # Species – α/anti – target   e.g., "rabbit–α-human von Willebrand Factor", "mouse-α-smooth muscle actin clone 1A4"
        \b(?:rabbit|mouse|goat|rat|sheep|hamster|donkey|chicken|guinea\s*pig|llama|alpaca|human)
          \s*(?:[–-]\s*)?(?:α|anti[-\s]?)
          \s*[^\),;]{2,80}                                   # up to a comma/paren/semicolon boundary
        |
        # Citation tail style A: Journal + volume + year + pages
        \b(?:[A-Z][a-z]+\.?\s?){1,6}
        [\s,;:]+
        \d{1,4}
        (?:\s*\(\s*\d{1,4}\s*\))?
        [\s,;:]+
        (?:19\d{2}|20\d{2}|21\d{2})
        [\s,;:]+
        \d{1,6}(?:[-–]\d{1,6})?
        |
        # Citation tail style B: Journal + year ; volume : pages
        \b(?:[A-Z][a-z]+\.?\s?){1,6}
        \s+
        (?:19\d{2}|20\d{2}|21\d{2})
        \s*;\s*
        \d{1,4}
        \s*:\s*
        \d{1,6}(?:[-–]\d{1,6})?
        |
        # ---- Donor tissue cues ----
        \b(?:obtained|purified|isolated|harvested|derived)\s+from\b
    """

    # How many word‐tokens on each side of the matched token to look at
    WINDOW: int = 12

    def __init__(self):
        # Compile species‐patterns and false‐context pattern once
        self.compile_patterns()

    def compile_patterns(self) -> None:
        """
        Compiles:
          - self._species_patterns: dict mapping each label to list of re.Pattern
          - self._false_context: compiled false‐context regex (VERBOSE, IGNORECASE)
        """
        self._species_patterns: dict[str, List[re.Pattern]] = {}
        for label in self.SPECIES_LABELS:
            patt_list = self.RAW_PATTERNS.get(label, [])
            self._species_patterns[label] = [
                re.compile(p, flags=re.IGNORECASE) for p in patt_list
            ]

        # Compile false‐context pattern with IGNORECASE and VERBOSE
        self._false_context: re.Pattern = re.compile(
            self.FALSE_CONTEXT_TERMS, flags=re.IGNORECASE | re.VERBOSE
        )

    def _find_first_match(self, regex_obj: re.Pattern, text: str) -> Optional[Match[str]]:
        """
        Return the first match or None.
        """
        return regex_obj.search(text)

    def _find_all_matches(self, regex_obj: re.Pattern, text: str) -> List[Match[str]]:
        """
        Return all non‐overlapping matches.
        """
        return list(regex_obj.finditer(text))

    def _is_in_false_context(
        self, text: str, match_start: int, match_end: int, window: int
    ) -> bool:
        """
        Returns True if ANY multi‐token or single‐token false‐context pattern
        (e.g. "isolated from", "antibody", "biotinylated", etc.) appears
        within +/- window WORD‐tokens of the matched token span [match_start, match_end).

        We do this by:
          1. Finding all word‐token spans in `text`.
          2. Locating which token(s) overlap the match range.
          3. Expanding that to +/- window tokens on each side.
          4. Extracting the substring of `text` that covers those tokens.
          5. Searching `self._false_context` against that entire snippet.

        If `self._false_context` matches anywhere in that snippet, return True.
        Otherwise, return False.
        """
        
        # 1. Tokenize entire text into word tokens (match objects)
        tokens = list(re.finditer(r"\b\w+\b", text))
        
        # 2. Find token indices overlapping [match_start, match_end)
        overlapping_indices = [
            i
            for i, tok in enumerate(tokens)
            if (tok.start() < match_end and tok.end() > match_start)
        ]
        if not overlapping_indices:
            return False

        # We'll check each overlapping token index. If ANY window around it
        # contains a false-context match, we return True immediately.
        for idx in overlapping_indices:
            # 3. Compute window boundaries in token space
            start_idx = max(0, idx - window)
            end_idx = min(len(tokens), idx + window + 1)

            # 4. Get the character‐level span in `text` covering all tokens [start_idx .. end_idx-1]
            char_start = tokens[start_idx].start()
            char_end = tokens[end_idx - 1].end()

            snippet = text[char_start:char_end]

            # 5. If any FALSE_CONTEXT_TERMS match anywhere in that snippet, return True
            try:
                if self._false_context.search(snippet, timeout=10.0):
                    return True
            except:
                # Treat as NOT in false context on timeout (fail-open),
                print("WARNING: Context processing timeout.")
                return False

        return False

    def classify(self, text: str) -> Tuple[List[int], List[str]]:
        vector: List[int] = [0] * len(self.SPECIES_LABELS)
        found_labels: set[str] = set()
        matched_spans: List[Tuple[int, int]] = []

        for idx, label in enumerate(self.SPECIES_LABELS):
            for regex_obj in self._species_patterns.get(label, []):
                for match_obj in regex_obj.finditer(text):
                    match_start, match_end = match_obj.start(), match_obj.end()

                    # Check for overlap with any previous match
                    if any(s <= match_start < e or s < match_end <= e for s, e in matched_spans):
                        continue
                    
                    if not self._is_in_false_context(text, match_start, match_end, self.WINDOW):
                        vector[idx] = 1
                        found_labels.add(label)
                        matched_spans.append((match_start, match_end))
                        break  # Stop after first valid match for this label
                if vector[idx] == 1:
                    break

        if not found_labels:
            other_idx = self.SPECIES_LABELS.index("species-other")
            vector[other_idx] = 1
            found_labels.add("species-other")

        return vector, list(found_labels)