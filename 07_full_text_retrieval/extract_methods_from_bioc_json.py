import json
import os
import pandas as pd

def extract_methods_subtitles_to_csv(json_path, output_csv, logs_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for doc in data[0].get("documents", []):
        passages = doc.get("passages", [])

        # Try to extract PMID from infons
        pmid = None
        for p in passages:
            pmid = p.get("infons", {}).get("article-id_pmid")
            if pmid:
                break
        if not pmid:
            # Fallback: extract from filename
            pmid = os.path.splitext(os.path.basename(json_path))[0]

        current_subtitle = "METHODS"
        in_methods = False

        for p in passages:
            section_type = p.get("infons", {}).get("section_type", "").upper()
            type_ = p.get("infons", {}).get("type", "").lower()
            text = p.get("text", "")

            if section_type == "METHODS":
                in_methods = True

                if type_ == "title_2":
                    current_subtitle = text.strip() or "METHODS"
                elif type_ == "paragraph":
                    rows.append({
                        "pmid": pmid,
                        "subtitle": current_subtitle,
                        "paragraph": text.strip()
                    })

            elif section_type != "METHODS":
                continue  # Allow non-METHODS to be skipped without stopping

    # Save to CSV
    if len(rows) > 0:
        df = pd.DataFrame(rows, columns=["pmid", "subtitle", "paragraph"])
        df.to_csv(output_csv, index=False)
        unique_subtitles = df["subtitle"].nunique()
        print(f"Saved {len(rows)} rows to {output_csv} with {unique_subtitles} unique subtitles")
        return True, unique_subtitles
    else:
        print(f"Skipped {len(rows)}")
        if logs_dir:
            with open(logs_dir + "no_methods_pmids.txt", "a") as f:
                f.write(f"{pmid}\n")
        return False, 0

def process_each_json_in_dir(json_dir, output_dir, logs_dir, disease):
    os.makedirs(output_dir, exist_ok=True)
    success = 0
    total = 0
    subtitle_counts = []

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            pmid = os.path.splitext(filename)[0]
            output_csv = os.path.join(output_dir, f"methods_subtitles_{pmid}.csv")

            saved, unique_subtitles = extract_methods_subtitles_to_csv(json_path, output_csv, logs_dir)
            success += 1 if saved else 0
            total += 1
            if saved:
                subtitle_counts.append(unique_subtitles)

    success_rate = (success / total) * 100 if total else 0
    avg_subtitles = sum(subtitle_counts) / len(subtitle_counts) if subtitle_counts else 0

    print(f"\nSummary Statistics {disease}:")
    print(f"  Total PMIDs processed    : {total}")
    print(f"  Successful methods extractions   : {success}")
    print(f"  Success rate             : {success_rate:.2f}%")
    print(f"  Avg. unique subtitles    : {avg_subtitles:.2f}")
    
    # Save summary stats
    summary_path = os.path.join(logs_dir, f"summary_stats_pmc_methods_{disease}.txt")
    with open(summary_path, "w") as f:
        f.write("\nSummary Statistics:\n")
        f.write(f"Total PMIDs processed  : {total}\n")
        f.write(f"Successful methods extractions : {success}\n")
        f.write(f"Success rate           : {success_rate:.2f}%\n")
        f.write(f"Avg. unique subtitles  : {avg_subtitles:.2f}")


def main():
    disease = "epilepsy"  # Change this to "parkinson" or "epilepsy" as needed
    print(f"Processing {disease} methods extraction from PMC JSON files...")
    process_each_json_in_dir(
        json_dir=f"07_full_text_retrieval/pmc_fulltext/{disease}_fulltext/",
        output_dir=f"07_full_text_retrieval/materials_methods/from_pmc_json/{disease}_methods/",
        logs_dir="07_full_text_retrieval/materials_methods/logs/",
        disease=disease
        )

if __name__ == "__main__":
    main()
