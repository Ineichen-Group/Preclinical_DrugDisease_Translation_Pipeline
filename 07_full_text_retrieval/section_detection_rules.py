import re

MATERIALS_METHODS_TITLES = [
    r"materials\s*(and|&)?\s*methods",
    r"materials",
    r"methodology",
    #r"experimental",
    r"experimental\s+(procedure[s]?|section[s]?)",
    r"methods",
]

STOP_SECTION_TITLES = [
    "RESULTS",
    "DISCUSSION",
    "CONCLUSION",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENT",
    "REFERENCES",
    "BIBLIOGRAPHY",
    "SUPPLEMENTARY MATERIALS",
    "SUPPORTING INFORMATION",
]


def is_start_of_materials_methods(text):
    pattern = re.compile(
        r"^\s*(\d+\.?|\b[IVXLCDM]+\b\.?)?\s*.*?\b(" + "|".join(MATERIALS_METHODS_TITLES) + r")\b",
        re.IGNORECASE
    )
    return bool(pattern.search(text.strip()))


def is_end_of_materials_methods(text):
    upper_text = text.strip().upper()
    return any(keyword in upper_text for keyword in STOP_SECTION_TITLES)
