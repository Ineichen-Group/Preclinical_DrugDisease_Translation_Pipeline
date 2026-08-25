#!/bin/bash
#SBATCH --job-name=preclin_fulltext_cadmus
#SBATCH --time=10:30:00
#SBATCH --output=cadmus_output_%A.log
#SBATCH --error=cadmus_error_%A.log
#SBATCH --mem=16G

# Add edirect directory to PATH
export PATH=${PATH}:/data/sdonev/cadmus/output/medline/edirect

PMID_FILE="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/07_full_text_retrieval/pmc_fulltext/logs/update_2025/failed_pmids_update_2025.txt"

PMID_COUNT=$(grep -cve '^[[:space:]]*$' "$PMID_FILE")
echo "Number of PMIDs: ${PMID_COUNT}"

# Start timer
start_time=$(date +%s)

echo "Running fetch_cadmus_fulltext.py..."
echo "PMID file: ${PMID_FILE}"

python fetch_cadmus_fulltext.py \
    --pmids "${PMID_FILE}"

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "fetch_cadmus_fulltext.py failed to execute."
    exit 1
else
    echo "fetch_cadmus_fulltext.py ran successfully."
fi

# End timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))

mins=$((elapsed / 60))
secs=$((elapsed % 60))

echo "Time elapsed: ${mins} minute(s) and ${secs} second(s)."