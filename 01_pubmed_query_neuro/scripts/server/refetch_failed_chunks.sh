#!/bin/bash
#SBATCH --job-name=refetch_pubmed_failed
#SBATCH --output=logs/refetch_pubmed_%A_%a.out
#SBATCH --error=logs/refetch_pubmed_%A_%a.err
#SBATCH --array=0-50%50       # CHANGE 99 to number_of_failed_chunks - 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07:00:00
#SBATCH --mem=2G

# Directory containing original PMID chunk files
INPUT_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/01_pubmed_query_neuro/data/results_pmids/update_2025/chunks"

# Output directory
OUTPUT_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/01_pubmed_query_neuro/data/full_pubmed_raw/update_2025"
mkdir -p "$OUTPUT_DIR"

# File containing chunks that need to be re-run
RETRY_FILE="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/01_pubmed_query_neuro/data/full_pubmed_raw/update_2025/chunk_files_under_4500.txt"

# Get the filename corresponding to this SLURM task
RETRY_CHUNK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RETRY_FILE")

if [ -z "$RETRY_CHUNK" ]; then
    echo "No chunk found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Retry entry: $RETRY_CHUNK"

# Extract chunk number from filename.
# Works with names such as:
#   pmid_chunk_023.txt
#   pmid_contents_chunk_23.txt
CHUNK_ID=$(basename "$RETRY_CHUNK" | grep -oE '[0-9]+' | tail -1)

if [ -z "$CHUNK_ID" ]; then
    echo "Could not extract chunk number from: $RETRY_CHUNK"
    exit 1
fi

# Reconstruct original input chunk filename
CHUNK_FILE=$(printf "%s/pmid_chunk_%03d.txt" "$INPUT_DIR" "$CHUNK_ID")

if [ ! -f "$CHUNK_FILE" ]; then
    echo "Chunk file not found: $CHUNK_FILE"
    exit 1
fi

# Create comma-separated PMID list
id_list=$(paste -sd, "$CHUNK_FILE")

# Output file - overwrite the incomplete/bad previous result
OUTPUT_FILE="$OUTPUT_DIR/pmid_contents_chunk_${CHUNK_ID}.txt"

MAX_ATTEMPTS=5

echo "Processing chunk $CHUNK_ID"
echo "Input:  $CHUNK_FILE"
echo "Output: $OUTPUT_FILE"

# Number of PMIDs expected
expected_rows=$(wc -l < "$CHUNK_FILE")

echo "Expected rows: $expected_rows"

for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do

    echo "Attempt $attempt..."

    efetch -db pubmed -id "$id_list" -format xml 2>> error.log | \
        xtract \
          -pattern PubmedArticle -tab "|||" -def "N/A" \
          -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
          -block PublicationTypeList -sep "+" -element PublicationType \
        > "$OUTPUT_FILE"

    actual_rows=$(wc -l < "$OUTPUT_FILE")

    if [[ "$actual_rows" -eq "$expected_rows" ]]; then
        echo "SUCCESS: chunk $CHUNK_ID has expected number of rows ($actual_rows)."
        exit 0
    else
        echo "Attempt $attempt: expected $expected_rows rows, got $actual_rows."

        if [[ $attempt -lt $MAX_ATTEMPTS ]]; then
            echo "Retrying in 15 seconds..."
            sleep 15
        else
            echo "FAILED: chunk $CHUNK_ID after $MAX_ATTEMPTS attempts."
            exit 1
        fi
    fi

done