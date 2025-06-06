import os
import time
import requests
import pandas as pd
import glob

# ---------- CONFIG ----------
EMAIL = "donevasimona@gmail.com"  # Your email for NCBI API access
FORMAT = "json"  # Choose "json" or "xml"

INPUT_DIR = "./06_preclin_clinic_join/data/preclinical_for_full_text"
OUTPUT_BASE_DIR = "07_full_text_retrieval/pmc_fulltext"
LOG_BASE_DIR = os.path.join(OUTPUT_BASE_DIR, "logs")

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOG_BASE_DIR, exist_ok=True)


# ---------- UTILITIES ----------
def is_pmc_open_access(pmcid):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        r = requests.get(url)
        return r.status_code == 200 and 'not-open-access' not in r.text
    except Exception as e:
        print(f"  Error checking OA status for {pmcid}: {e}")
    return False


def pmid_to_pmcid(pmid):
    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        f"?tool=fulltext-fetcher&email={EMAIL}&ids={pmid}&format=json"
    )
    try:
        r = requests.get(url)
        data = r.json()
        records = data.get("records", [])
        if records and "pmcid" in records[0]:
            return records[0]["pmcid"]
    except Exception as e:
        print(f"  Error converting PMID {pmid}: {e}")
    return None


def get_bioc_fulltext(pmcid, format="json"):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{pmcid}/unicode"
    r = requests.get(url)
    if r.status_code == 200 and "No result can be found" not in r.text:
        return r.text
    return None


def get_pmc_oai_xml(pmcid):
    pmcid_num = pmcid.replace("PMC", "")
    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?"
        f"verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc"
    )
    r = requests.get(url)
    if r.status_code == 200 and "cannotDisseminateFormat" not in r.text:
        return r.text
    return None


# ---------- MAIN FUNCTION ----------
def fetch_fulltexts(pmids, disease_name, save_dir, logs_dir, format="json"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    results = {}
    failed_pmids = []

    start_time = time.time()

    for i, pmid in enumerate(pmids, 1):
        print(f"[{i}/{len(pmids)}] Processing PMID: {pmid}")
        file_ext = format if format == "json" else "xml"
        filename = os.path.join(save_dir, f"{pmid}.{file_ext}")

        if os.path.exists(filename):
            print(f"  Skipping {pmid} — already downloaded")
            results[pmid] = filename
            continue

        pmcid = pmid_to_pmcid(pmid)
        if not pmcid:
            print("  No PMCID found.")
            failed_pmids.append(pmid)
            continue

        if not is_pmc_open_access(pmcid):
            print(f"  {pmcid} is not Open Access.")
            failed_pmids.append(pmid)
            continue

        print(f"  Found PMCID: {pmcid}, trying BioC {format.upper()}...")
        fulltext = get_bioc_fulltext(pmcid, format=format)

        if not fulltext:
            print("  BioC not available. Trying OAI XML...")
            fulltext = get_pmc_oai_xml(pmcid)
            file_ext = "xml"

        if fulltext:
            filename = os.path.join(save_dir, f"{pmid}.{file_ext}")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(fulltext)
            results[pmid] = filename
            print(f"  Full text saved to {filename}")
        else:
            print("  Full text not available from any source.")
            failed_pmids.append(pmid)

        time.sleep(0.5)  # Be polite to the API

    # Save failed PMIDs with disease in filename
    failed_filename = f"failed_pmids_{disease_name}.txt"
    failed_path = os.path.join(logs_dir, failed_filename)
    with open(failed_path, "w") as f:
        f.writelines(f"{pmid}\n" for pmid in failed_pmids)

    # Save summary stats with disease in filename
    summary_filename = f"summary_{disease_name}.txt"
    summary_path = os.path.join(logs_dir, summary_filename)

    total = len(pmids)
    success = len(results)
    elapsed = time.time() - start_time

    with open(summary_path, "w") as f:
        f.write(f"Disease       : {disease_name}\n")
        f.write(f"Total PMIDs   : {total}\n")
        f.write(f"Successful    : {success}\n")
        f.write(f"Failed        : {len(failed_pmids)}\n")
        f.write(f"Time taken    : {elapsed:.2f} seconds\n")

    print(f"\nSummary saved to '{summary_path}'")
    print(f"Failed PMIDs saved to '{failed_path}'")
    print(f"Completed {success}/{total} for {disease_name} in {elapsed:.2f} seconds")

    return results


# ---------- RUN FOR ALL DISEASE FILES ----------
if __name__ == "__main__":
    disease_files = glob.glob(f"{INPUT_DIR}/*_pmids.csv")

    for disease_file in disease_files:
        disease_name = os.path.basename(disease_file).replace("_pmids.csv", "")
        if disease_name != "epilepsy":
            print(f"Skipping {disease_name} as it is not epilepsy.")
            continue
        save_dir = os.path.join(OUTPUT_BASE_DIR, f"{disease_name}_fulltext")
        logs_dir = os.path.join(LOG_BASE_DIR, disease_name)

        try:
            pmid_list = pd.read_csv(disease_file)['PMID'].astype(str).tolist()
            print(f"\n==============================")
            print(f"Processing disease: {disease_name} with {len(pmid_list)} PMIDs")
            print(f"==============================")
            
            fetch_fulltexts(pmid_list, disease_name, save_dir, logs_dir, format=FORMAT)
        except Exception as e:
            print(f"Error processing {disease_name}: {e}")
