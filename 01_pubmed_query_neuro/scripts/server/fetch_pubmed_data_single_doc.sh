#!/bin/bash
#SBATCH --job-name=fetch_pubmed_data_single      # Job name
#SBATCH --output=logs/fetch_pubmed_data_test.out    # Standard output and error log
#SBATCH --error=logs/fetch_pubmed_data_test.err     # Error log
#SBATCH --mem=4G

# Test PubMed Data Fetch Script

# Ensure output directory exists
OUTPUT_DIR="./pubmed_results"
mkdir -p "$OUTPUT_DIR"

# Use the test chunk file
CHUNK_FILE="pmid_chunk_188.txt"

# Check if CHUNK_FILE exists
if [ ! -f "$CHUNK_FILE" ]; then
    echo "Test chunk file not found!"
    exit 1
fi

# Create a comma-separated list of PMIDs from the chunk file
id_list=$(paste -sd, "$CHUNK_FILE")

# Set output file for the test
OUTPUT_FILE="$OUTPUT_DIR/pmid_contents_test_39596651.txt"

# Fetch data and save to OUTPUT_FILE
echo "Processing test file..."
efetch -db pubmed -id 39596651 -format xml | \
    xtract -pattern PubmedArticle -tab "|||" -def "N/A" \
    -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
    -block PublicationTypeList -sep "+" -element PublicationType \
    > "$OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
    echo "Finished processing test file successfully. Results saved to $OUTPUT_FILE"
else
    echo "Failed to fetch data or output file is empty."
fi
