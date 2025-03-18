#!/bin/bash
#SBATCH --job-name=fetch_pubmed_data_large      # Job name
#SBATCH --output=logs/fetch_pubmed_data_%A_%a.out    # Standard output and error log
#SBATCH --error=logs/fetch_pubmed_data_%A_%a.err     # Error log
#SBATCH --array=0-5000%50                       # Array range, 50 tasks concurrently
#SBATCH --ntasks=1                              # Number of tasks per chunk
#SBATCH --cpus-per-task=1                       # Number of CPU cores per task
#SBATCH --time=05:00:00                         # Time limit hrs:min:sec
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
    -block ArticleId -if ArticleId@IdType -equals doi -element ArticleId \
    -block PublicationTypeList -sep "+" -element PublicationType | \
    awk -F '\\|\\|\\|' '
    BEGIN {
        OFS = "|||"
    }
    {
        ################################################################
        # 1) If the last field contains a tab, split it into two fields.
        #    (Sometimes xtract lumps "DOI<TAB>PublicationType" together.)
        ################################################################
        n = split($NF, arr, /\t/)
        if (n == 2) {
            $NF = arr[1]          # e.g. the actual DOI
            $(NF + 1) = arr[2]    # e.g. "Journal Article" (or other pub type)
        }

        ################################################################
        # 2) If we only have 6 fields total, that means the DOI field
        #    never appeared at all. Insert "N/A" as the 6th field,
        #    pushing the last field (pub type) to 7th.
        ################################################################
        if (NF == 6) {
            last_field = $NF
            $NF = "N/A"           # new 6th field
            $(NF + 1) = last_field   # push the original 6th field to 7th
        }

        print
    }' \
    > "$OUTPUT_FILE"

echo "Finished processing $CHUNK_FILE"
