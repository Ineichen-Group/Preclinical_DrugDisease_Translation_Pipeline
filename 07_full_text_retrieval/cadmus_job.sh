#!/bin/bash
#SBATCH --job-name=preclin_fulltext_cadmus
#SBATCH --time=10:30:00
#SBATCH --output=cadmus_output_%A.log
#SBATCH --error=cadmus_error_%A.log
#SBATCH --mem=16G

BASE_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/07_full_text_retrieval"

SCRIPT="${BASE_DIR}/fetch_cadmus_fulltext.py"
PMID_FILE="${BASE_DIR}/pmc_fulltext/logs/update_2025/failed_pmids_update_2025.txt"
API_KEYS="${BASE_DIR}/api_keys.txt"

CADMUS_RUN_DIR="${BASE_DIR}/cadmus_update_2025"

mkdir -p "${CADMUS_RUN_DIR}"

PMID_COUNT=$(grep -cve '^[[:space:]]*$' "${PMID_FILE}")

echo "Number of PMIDs: ${PMID_COUNT}"
echo "PMID file: ${PMID_FILE}"
echo "CADMUS run directory: ${CADMUS_RUN_DIR}"

# CADMUS creates ./output relative to the current working directory
cd "${CADMUS_RUN_DIR}" || exit 1

# Start timer
start_time=$(date +%s)

echo "Running fetch_cadmus_fulltext.py..."

python -u "${SCRIPT}" \
    --pmids "${PMID_FILE}" \
    --api_keys "${API_KEYS}"

status=$?

if [ "${status}" -ne 0 ]; then
    echo "fetch_cadmus_fulltext.py failed with exit code ${status}."
    exit "${status}"
else
    echo "fetch_cadmus_fulltext.py ran successfully."
fi

# End timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))

mins=$((elapsed / 60))
secs=$((elapsed % 60))

echo "Time elapsed: ${mins} minute(s) and ${secs} second(s)."