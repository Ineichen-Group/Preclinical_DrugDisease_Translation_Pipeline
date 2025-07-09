from text_to_num import text2num
import re
import pandas as pd
from typing import Union, List, Optional


def parse_fragment(fragment: str) -> Optional[float]:
    """
    Attempt to parse a single "number fragment" (no commas) into a float.
    Returns:
        • float if successful
        • None if parsing fails
    """
    frag = fragment.strip().lower()

    # Remove common prefixes like 'n ='
    frag = re.sub(r"^n\s*=\s*", "", frag)

    # Remove ordinal suffixes (e.g., "1st" -> "1")
    frag = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", frag)

    # Remove leading/trailing noise characters (non-alphanumeric)
    frag = re.sub(r"^[^0-9a-z]+", "", frag)
    frag = re.sub(r"[^0-9a-z]+$", "", frag)
    if not frag:
        return None

    # If purely numeric (integer or decimal)
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", frag):
        try:
            return float(frag)
        except ValueError:
            return None

    # Attempt to parse spelled-out words
    try:
        return float(text2num(frag, "en"))
    except ValueError:
        return None


def normalize_number(s: str) -> Union[float, List[float], str]:
    """
    Convert a string `s`—which may be:
        • Purely numeric ("123", "45.6")
        • Spelled‐out English words ("one hundred twenty-three", "forty-five point six")
        • Multiple comma‐separated values (e.g., "Four, Twenty")
        • Contain leading noise characters (e.g., "= 46")

    Returns:
        • float: if a single value is successfully parsed
        • List[float]: if multiple comma‐separated fragments all parse successfully
        • str: the original input if any fragment cannot be parsed

    Steps:
        1. Strip leading/trailing whitespace and lowercase the input.
        2. Remove ordinal suffixes (e.g., "1st" → "1").
        3. Remove leading noise characters (non-alphanumeric, e.g., "=", "<", "~").
        4. If the cleaned string contains commas, split on commas and parse each fragment:
            a. If all fragments parse to floats, return a list of floats.
            b. Otherwise, log a warning and return the original string.
        5. If no commas:
            a. Call `parse_fragment` on the cleaned string.
            b. If parsing succeeds, return the float.
            c. Otherwise, log a warning and return the original string.

    Args:
        s (str): The input string representing one or more numbers.

    Returns:
        Union[float, List[float], str]: Parsed number(s) or original string if parsing fails.
    """
    # 1) Normalize whitespace & case
    s_clean = s.strip().lower()

    # 2) Remove ordinal suffixes from the entire string
    s_clean = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s_clean)

    # 3) Remove leading noise from the entire string
    s_clean = re.sub(r"^[^0-9a-z]+", "", s_clean)
    
    # 4) Remove leading "a " if followed by a number word (e.g., "a hundred")
    s_clean = re.sub(r"^a\s+(?=\w+)", "", s_clean)

    # 5) If commas are present, split and parse each part
    if "," in s_clean:
        parts = s_clean.split(",")
        parsed_list: List[float] = []
        for part in parts:
            parsed_value = parse_fragment(part)
            if parsed_value is None:
                print(f"Warning: Could not parse '{part.strip()}' as a number. Returning original value.")
                return s  # Return the original unmodified input
            parsed_list.append(parsed_value)
        parsed_list_unique = list(set(parsed_list))
        if len(parsed_list_unique) == 1:
            return parsed_list_unique[0]
        return parsed_list

    # 5) Single value (no commas)
    single_parsed = parse_fragment(s_clean)
    if single_parsed is not None:
        return single_parsed

    # 6) If parsing failed
    print(f"Warning: Could not parse '{s}' as a number. Returning original value.")
    return s

if __name__ == "__main__":
    # 1) Read the CSV file containing the `prediction_encoded_label` column
    input_path = "./08_IE_full_text/model_predictions/animals_nr/doc_animals_nr_predictions_clean.csv"
    df = pd.read_csv(input_path)

    # 2) Drop rows where `prediction_encoded_label` is NaN or empty
    df_clean = df.dropna(subset=["prediction_encoded_label"]).copy()
    df_clean["prediction_encoded_label_raw"] = (
        df_clean["prediction_encoded_label"].astype(str).str.strip()
    )
    df_clean = df_clean[df_clean["prediction_encoded_label_raw"] != ""]

    # 3) Apply normalization to each label
    df_clean["prediction_encoded_label"] = df_clean["prediction_encoded_label_raw"].apply(normalize_number)

    # 4) (Optional) If you want to drop any rows where normalization returned a non‐float/list,
    #    uncomment the following lines to keep only numeric results:
    #    df_clean = df_clean[
    #        df_clean["numeric_label"].apply(lambda x: isinstance(x, (float, list)))
    #    ]

    # 6) Save to a new CSV (will contain floats or lists in `numeric_label`)
    output_path = "./08_IE_full_text/model_predictions/animals_nr/animals_nr_predictions_numeric.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved converted labels to: {output_path}")





