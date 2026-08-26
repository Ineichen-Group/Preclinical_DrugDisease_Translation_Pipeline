#!/bin/bash
#SBATCH --job-name=umls_map_parent
#SBATCH --output=logs/umls_map_parent_%j.out
#SBATCH --error=logs/umls_map_parent_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---- script ----
SCRIPT_PATH="umls_map_to_parent.py"

# ---- UMLS resources ----
MRREL_PATH="./data/umls/mrrel_all_drug_rela_20251209.csv"
ID_TO_TERM_MAP="./data/umls/umls_id_to_term_map.json"

# ---- reference inputs ----

# Historical clinical data:
# contributes candidate UMLS parent IDs.
CLINICAL_REFERENCE_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/clinical/drug_mapped_clinical_drug_enriched.csv"

# Historical preclinical data:
# also contributes candidate UMLS parent IDs.
PRECLINICAL_REFERENCE_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/drug_mapped_preclinical_enriched_all.csv"

# ---- new preclinical dataset ----

PRECLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/drug_mapped_preclinical_enriched_all_update_2025.csv"

# ---- outputs ----

PRECLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_drug_data_with_umls_parents_update_2025.csv"

STATS_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/umls/umls_mapped_to_parents_preclinical_update_2025_stats.csv"

mkdir -p \
    logs \
    "$(dirname "$PRECLINICAL_OUTPUT")" \
    "$(dirname "$STATS_OUTPUT")"

echo "===== JOB INFO ====="
echo "Job ID:                        ${SLURM_JOB_ID:-local}"
echo "Node:                          ${SLURMD_NODENAME:-local}"
echo "CPUs:                          ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "Clinical reference:            $CLINICAL_REFERENCE_INPUT"
echo "Preclinical reference:         $PRECLINICAL_REFERENCE_INPUT"
echo "New preclinical input:         $PRECLINICAL_INPUT"
echo
echo "Preclinical output:            $PRECLINICAL_OUTPUT"
echo "Stats output:                  $STATS_OUTPUT"
echo "===================="

START_TIME=$(date +%s)

python "$SCRIPT_PATH" \
    --mrrel_path "$MRREL_PATH" \
    --id_to_term_map_path "$ID_TO_TERM_MAP" \
    --clinical_reference_input "$CLINICAL_REFERENCE_INPUT" \
    --preclinical_reference_input "$PRECLINICAL_REFERENCE_INPUT" \
    --preclinical_input "$PRECLINICAL_INPUT" \
    --preclinical_output "$PRECLINICAL_OUTPUT" \
    --stats_output "$STATS_OUTPUT"

STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ "$STATUS" -ne 0 ]; then
    echo "UMLS parent mapping failed with exit code $STATUS"
    exit "$STATUS"
fi

echo "Finished assigning UMLS parents in ${DURATION} seconds"