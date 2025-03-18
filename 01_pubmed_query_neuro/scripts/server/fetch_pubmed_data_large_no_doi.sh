#!/bin/bash
#SBATCH --job-name=fetch_pubmed_data_large      # Job name
#SBATCH --output=logs/fetch_pubmed_data_%A_%a.out    # Standard output and error log
#SBATCH --error=logs/fetch_pubmed_data_%A_%a.err     # Error log
#SBATCH --array=0-5000%50                       # Array range, 50 tasks concurrently
#SBATCH --ntasks=1                              # Number of tasks per chunk
#SBATCH --cpus-per-task=1                       # Number of CPU cores per task
#SBATCH --time=07:00:00                         # Time limit hrs:min:sec
#SBATCH --mem=2G                                # Memory per task

# Ensure output directory exists
OUTPUT_DIR="./pubmed_results"
mkdir -p "$OUTPUT_DIR"

# Select the chunk file for this task based on SLURM_ARRAY_TASK_ID
CHUNK_FILE=$(printf "pmid_chunk_%03d.txt" "$SLURM_ARRAY_TASK_ID")

# Check if CHUNK_FILE exists
if [ ! -f "$CHUNK_FILE" ]; then
    echo "Chunk file not found for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Create a comma-separated list of PMIDs from the chunk file
id_list=$(paste -sd, "$CHUNK_FILE")

# Set output file for this chunk
OUTPUT_FILE="$OUTPUT_DIR/pmid_contents_chunk_${SLURM_ARRAY_TASK_ID}.txt"

# Fetch data and save to OUTPUT_FILE
echo "Processing $CHUNK_FILE..."
efetch -db pubmed -id "$id_list" -format xml | \
    xtract -pattern PubmedArticle -tab "|||" -def "N/A" \
    -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
    -block PublicationTypeList -sep "+" -element PublicationType \
    > "$OUTPUT_FILE"

echo "Finished processing $CHUNK_FILE"
