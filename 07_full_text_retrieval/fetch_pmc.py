import os
import time
import requests
import pandas as pd

# ---------- CONFIG ----------
EMAIL = "donevasimona@gmail.com"  # Replace with your real email
FORMAT = "json"  # Choose "json" or "xml"
SAVE_DIR = "07_full_text_retrieval/pmc_fulltext/MS"
LOGS_DIR = "07_full_text_retrieval/pmc_fulltext/logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def is_pmc_open_access(pmcid):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
    try:
        r = requests.get(url)
        if r.status_code == 200 and 'not-open-access' not in r.text:
            return True
    except Exception as e:
        print(f"  ⚠️ Error checking OA status for {pmcid}: {e}")
    return False


# ---------- CONVERTERS ----------
def pmid_to_pmcid(pmid):
    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        f"?tool=fulltext-fetcher&email={EMAIL}&ids={pmid}&format=json"
    )
    try:
        r = requests.get(url)
        data = r.json()
        records = data.get("records", [])
        if records:
            record = records[0]
            if "pmcid" in record:
                return record["pmcid"]
            else:
                print(f"  PMID {pmid} found, but no PMCID available (not in PMC?)")
        else:
            print(f"  No records found for PMID {pmid}")
    except Exception as e:
        print(f"  Error converting PMID {pmid}: {e}")
    return None

# ---------- FETCHERS ----------
def get_bioc_fulltext(pmcid, format="json"):
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_{format}/{pmcid}/unicode"
    r = requests.get(url)
    if r.status_code == 200 and r.text.strip():
        if "No result can be found" in r.text:
            print("  BioC API response indicates no result found.")
            return None
        return r.text
    return None

def get_pmc_oai_xml(pmcid):
    pmcid_num = pmcid.replace("PMC", "")
    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?"
        f"verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid_num}&metadataPrefix=pmc"
    )
    r = requests.get(url)
    if r.status_code == 200 and r.text.strip():
        if "cannotDisseminateFormat" in r.text:
            print(" OAI response indicates unsupported format (not available as PMC full text XML).")
            return None
        return r.text
    return None


# ---------- MAIN FUNCTION ----------
def fetch_fulltexts(pmids, format="json"):
    results = {}
    failed_pmids = []

    start_time = time.time()

    for i, pmid in enumerate(pmids, 1):
        print(f"[{i}/{len(pmids)}] Processing PMID: {pmid}")

        file_ext = format if format == "json" else "xml"
        filename = os.path.join(SAVE_DIR, f"{pmid}.{file_ext}")

        if os.path.exists(filename):
            print(f"  Skipping {pmid} — already downloaded at {filename}")
            results[pmid] = filename
            continue

        pmcid = pmid_to_pmcid(pmid)
        if not pmcid:
            print("  No PMCID found.")
            failed_pmids.append(pmid)
            continue
        
        if not is_pmc_open_access(pmcid):
            print(f"  ⚠️ {pmcid} is not in PMC Open Access subset — skipping.")
            failed_pmids.append(pmid)
            continue

        print(f"  Trying BioC {format.upper()} for {pmcid}...")
        fulltext = get_bioc_fulltext(pmcid, format=format)

        if not fulltext:
            print("  BioC not available. Trying OAI XML fallback...")
            fulltext = get_pmc_oai_xml(pmcid)
            file_ext = "xml"

        if fulltext:
            filename = os.path.join(SAVE_DIR, f"{pmid}.{file_ext}")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(fulltext)
            results[pmid] = filename
            print(f"  Full text saved as {filename}")
        else:
            print("  Full text not available from any source.")
            failed_pmids.append(pmid)

        time.sleep(0.5)  # Be polite to the API

    # Save failed PMIDs
    failed_path = os.path.join(LOGS_DIR, "failed_pmids.txt")
    if failed_pmids:
        with open(failed_path, "w") as f:
            f.writelines(f"{pmid}\n" for pmid in failed_pmids)

    # Summary stats
    total = len(pmids)
    success = len(results)
    failed = len(failed_pmids)
    success_rate = (success / total) * 100 if total else 0

    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_minutes = elapsed / 60
    
    # Count how many were saved as JSON vs XML
    json_count = sum(1 for path in results.values() if path.endswith(".json"))
    xml_count = sum(1 for path in results.values() if path.endswith(".xml"))

    json_percent = (json_count / total) * 100 if total else 0
    xml_percent = (xml_count / total) * 100 if total else 0

    print("\nSummary Statistics:")
    print(f"  Total PMIDs processed: {total}")
    print(f"  Successful downloads  : {success}")
    print(f"  Failed downloads      : {failed}")
    print(f"  Success rate          : {success_rate:.2f}%")
    print(f"  ✅ % BioC JSON         : {json_percent:.2f}%")
    print(f"  📄 % OAI XML fallback  : {xml_percent:.2f}%")
    print(f"  Time taken            : {elapsed:.2f} seconds ({elapsed_minutes:.2f} minutes)")

    if failed:
        print(f"  Failed PMIDs saved to '{failed_path}'")

    # Save summary stats
    summary_path = os.path.join(LOGS_DIR, "summary_stats.txt")
    with open(summary_path, "w") as f:
        f.write("Summary Statistics:\n")
        f.write(f"Total PMIDs processed: {total}\n")
        f.write(f"Successful downloads : {success}\n")
        f.write(f"Failed downloads     : {failed}\n")
        f.write(f"Success rate         : {success_rate:.2f}%\n")
        f.write(f"% BioC JSON          : {json_percent:.2f}%\n")
        f.write(f"% OAI XML fallback   : {xml_percent:.2f}%\n")
        f.write(f"Time taken           : {elapsed:.2f} seconds ({elapsed_minutes:.2f} minutes)\n")


    print(f"  Summary stats saved to '{summary_path}'")

    return results

# ---------- RUN ----------
if __name__ == "__main__":
    pmid_list = pd.read_csv("03_ner/data/animal_studies_with_drug_disease/filtered_df_non_empty_MS_5380_PMIDs.csv")['PMID'].astype(str).tolist()
    fetch_fulltexts(pmid_list, format=FORMAT)
