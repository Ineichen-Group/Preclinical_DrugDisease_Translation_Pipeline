import re

MATERIALS_METHODS_TITLES = [
    r"materials\s*(and|&)?\s*methods",           # matches "Materials and Methods"
    r"materials",                                # matches just "Materials"
    r"methodology",                              # matches "Methodology"
    r"experimental\s+(procedure[s]?|section[s]?)",  # matches "Experimental Procedures" or "Experimental Sections"
    r"method[s]?",                               # matches "Method" or "Methods"
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
    text = text.strip().lower()
    for pattern in MATERIALS_METHODS_TITLES:
        if re.search(pattern, text):
            return True
    return False


def is_end_of_materials_methods(text):
    upper_text = text.strip().upper()
    return any(keyword in upper_text for keyword in STOP_SECTION_TITLES)
