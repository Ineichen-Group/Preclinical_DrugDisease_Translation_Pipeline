# utils/format_utils.py

from typing import List, Tuple, Any

def vector_to_str(vector: List[int]) -> str:
    """
    Turn a list of integers (e.g. [0,1,0,1]) into a comma-separated string "0,1,0,1".
    """
    return ",".join(str(v) for v in vector)

def labels_to_str(labels: List[str]) -> str:
    """
    Turn a list of strings (e.g. ['mouse','rat']) into a repr string "['mouse', 'rat']".
    If you later prefer JSON, you could do `json.dumps(labels)` here instead.
    """
    return str(labels)

def format_species_result(
    result: Tuple[List[int], List[str]]
) -> Tuple[str, str]:
    """
    Given the tuple that SpeciesClassifier.classify(text) returns—
    i.e. ([0,1,0,...], ['mouse','rat',...])—produce:
      - prediction_encoded_num:  "0,1,0,..."
      - prediction_encoded_label: "['mouse', 'rat']"
    """
    vector, labels = result
    return vector_to_str(vector), labels_to_str(labels)

def format_generic_result(result: Any) -> Tuple[Any, Any]:
    """
    A fallback wrapper for any classifier that already returns exactly
    (encoded_num, encoded_label). In that case, just pass through.
    """
    return result, None
