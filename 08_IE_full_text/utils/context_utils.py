# utils/context_utils.py

import re

FALSE_CONTEXT_TERMS = r"""
    \b(
        antibody|antibodies|antiserum|
        monoclonal|polyclonal|
        Ig\s*[A-Z]{1,2}|
        mAb|pAb|HRP|APC|FITC|PE|Cy\d+|
        ELISA|immunoblot|western blot|immunostaining|
        conjugated|biotinylated|fluorescent-labeled|
        GAPDH|tubulin|β-actin|
        luciferase|peroxidase|polymerase|qPCR|RT-PCR|Taq|
        serum|lysate|recombinant|TG2|anti|OX\d+|CD\d+
    )\b
"""

# How many word-tokens on each side of the match to look at
WINDOW = 10

def is_in_false_context(
    text: str,
    match_start: int,
    match_end: int,
    context_terms_regex: str = FALSE_CONTEXT_TERMS,
    window: int = WINDOW,
) -> bool:
    """
    Return True if any token within +/- window of the matched token
    matches one of the context_terms_regex. Otherwise False.
    """
    # 1) Tokenize entire text into word tokens
    tokens = list(re.finditer(r"\b\w+\b", text))

    # 2) Find which token indices overlap the character span [match_start, match_end)
    overlapping_indices = [
        i for i, tok in enumerate(tokens) if match_start <= tok.start() < match_end
    ]
    if not overlapping_indices:
        return False

    # 3) For each overlapping token index, look +/- window tokens
    for idx in overlapping_indices:
        start_idx = max(0, idx - window)
        end_idx = min(len(tokens), idx + window + 1)
        for tok_obj in tokens[start_idx:end_idx]:
            if re.search(context_terms_regex, tok_obj.group(), re.IGNORECASE | re.VERBOSE):
                return True
    return False
