#!/bin/bash
#SBATCH --job-name=fetch_pubmed_data_large      # Job name
#SBATCH --output=logs/missing/fetch_pubmed_data_%A_%a.out    # Standard output and error log
#SBATCH --error=logs/missing/fetch_pubmed_data_%A_%a.err     # Error log
#SBATCH --array=1-297                        # Large array with concurrent limit
#SBATCH --ntasks=1                              # Number of tasks per chunk
#SBATCH --cpus-per-task=1                       # Number of CPU cores per task
#SBATCH --time=07:00:00                         # Time limit hrs:min:sec
#SBATCH --mem=2G                                # Memory per task

# Read list of chunk files
CHUNK_FILES=($(cat chunks_missing_content.txt))

# Select the chunk file based on the job array index
CURRENT_CHUNK_FILE=${CHUNK_FILES[$((SLURM_ARRAY_TASK_ID-1))]}

# Ensure output directory exists
OUTPUT_DIR="./pubmed_results/missing_chunks_data"
mkdir -p "$OUTPUT_DIR"

# Check if chunk file exists
if [ ! -f "$CURRENT_CHUNK_FILE" ]; then
    echo "Chunk file not found: $CURRENT_CHUNK_FILE"
    exit 1
fi

# Create a comma-separated list of PMIDs from the chunk file
id_list=$(paste -sd, "$CURRENT_CHUNK_FILE")

# Set output file for this chunk
OUTPUT_FILE="$OUTPUT_DIR/processed_$(basename "$CURRENT_CHUNK_FILE")"

# Fetch data and save to OUTPUT_FILE
echo "Processing $CURRENT_CHUNK_FILE..."
efetch -db pubmed -id "$id_list" -format xml | \
    xtract -pattern PubmedArticle -tab "|||" -def "N/A" \
    -element MedlineCitation/PMID PubDate/Year Journal/Title ArticleTitle AbstractText \
    -block PublicationTypeList -sep "+" -element PublicationType \
    > "$OUTPUT_FILE"

echo "Finished processing $CURRENT_CHUNK_FILE"