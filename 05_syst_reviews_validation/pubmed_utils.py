import requests
from time import sleep
from xml.etree import ElementTree as ET
from rapidfuzz.distance import Levenshtein

def get_pmids_for_title(title: str, email: str = "example@example.com", api_key: str = None) -> list[str] | None:
    """
    Query PubMed and return a list of PMIDs of articles based on an exact title match.

    Args:
        title (str): Exact title of the article.
        email (str): Contact email (required by NCBI).
        api_key (str | None): NCBI API key (optional).

    Returns:
        list[str] | None: List of matching PMIDs or None if none found.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "email": email,
        "api_key": api_key,
        "retmode": "json",
        "db": "pubmed",
        "retmax": "1",
        "term": f'"{title}"[Title:~0]',
    }
    response = requests.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    results = response.json().get("esearchresult", {})
    ids = results.get("idlist", [])
    return ids if ids else None


def get_pmids_for_doi(doi: str) -> list[str]:
    """
    Query PubMed and return a list of PMIDs based on DOI.

    Args:
        doi (str): DOI string.

    Returns:
        list[str]: List of matching PMIDs (may be empty).
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": f"{doi}[DOI]", "retmode": "json"}
    response = requests.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    data = response.json().get("esearchresult", {})
    return data.get("idlist", [])


def get_title_for_pmid(pmid: str) -> str | None:
    """
    Retrieve the article title for a given PMID via PubMed efetch.

    Args:
        pmid (str): PubMed ID.

    Returns:
        str | None: Article title or None if not retrievable.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    response = requests.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    xml_root = ET.fromstring(response.content)
    article = xml_root.find('.//ArticleTitle')
    if article is None:
        return None
    # Safely extract text, including nested tags if any
    text = ''.join(article.itertext() or []).strip()
    return text if text else None


def is_title_match(expected: str, actual: str, threshold: int = 90) -> bool:
    """
    Check whether two titles match based on normalized Levenshtein similarity.

    Args:
        expected (str): Expected title.
        actual (str): Retrieved title.
        threshold (int): Similarity threshold (0-100).

    Returns:
        bool: True if similarity >= threshold.
    """
    score = Levenshtein.normalized_similarity(expected, actual) * 100
    return score >= threshold
