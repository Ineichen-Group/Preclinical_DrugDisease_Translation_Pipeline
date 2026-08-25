import argparse

from cadmus import bioscraping


def load_api_keys(filepath):
    api_keys = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            api_keys[key.strip()] = value.strip()

    return api_keys


def load_pmids(path_to_pmids):
    with open(path_to_pmids, "r") as f:
        pmid_list = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Remove duplicates while preserving order
    return list(dict.fromkeys(pmid_list))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch full text for a list of PMIDs using CADMUS."
    )

    parser.add_argument(
        "--pmids",
        required=True,
        help="Path to a TXT file containing one PMID per line.",
    )

    parser.add_argument(
        "--api_keys",
        default="api_keys.txt",
        help="Path to the API keys file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pmid_list = load_pmids(args.pmids)

    print(f"Will be fetching {len(pmid_list)} PMIDs.")

    keys = load_api_keys(args.api_keys)

    wiley_api_key_uoz = keys.get("wiley_api_key_uoz")
    elsevier_api_key_uoz = keys.get("elsevier_api_key_uoz")
    ncbi_api_key = keys.get("NCBI_API_KEY")

    bioscraping(
        pmid_list,
        "donevasimona@gmail.com",
        ncbi_api_key,
        wiley_api_key=wiley_api_key_uoz,
        elsevier_api_key=elsevier_api_key_uoz,
    )


if __name__ == "__main__":
    main()