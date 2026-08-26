#!/bin/bash
#SBATCH --job-name=mondo_map_parent
#SBATCH --output=logs/mondo_map_parent_%j.out
#SBATCH --error=logs/mondo_map_parent_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---- script ----
SCRIPT_PATH="mondo_map_to_parent.py"

# ---- reference inputs ----

# Historical clinical data:
# contributes possible MONDO parent nodes.
CLINICAL_REFERENCE_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_data_mondo_cleaned.csv"

# Historical preclinical data:
# also contributes possible MONDO parent nodes.
PRECLINICAL_REFERENCE_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_mondo_cleaned.csv"

# ---- new dataset to process ----

PRECLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_mondo_cleaned_update_2025.csv"

# ---- output ----

PRECLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_mondo_cleaned_with_mondo_parents_update_2025.csv"

STATS_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mondo/mondo_mapped_to_parents_preclinical_update_2025_stats.csv"

# ---- ontology + params ----

ONTOLOGY_PATH="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mondo/mondo.owl"
ROOT_ID="MONDO:0000001"

ID_COLUMN="disease_termid_mondo_clean"
LABEL_COLUMN="disease_term_mondo_clean"

MIN_DEPTH=5
MAX_DESC=20

mkdir -p \
    logs \
    "$(dirname "$PRECLINICAL_OUTPUT")" \
    "$(dirname "$STATS_OUTPUT")"

echo "===== JOB INFO ====="
echo "Job ID:                       ${SLURM_JOB_ID:-local}"
echo "Node:                         ${SLURMD_NODENAME:-local}"
echo "CPUs:                         ${SLURM_CPUS_PER_TASK:-1}"
echo
echo "Clinical reference:           $CLINICAL_REFERENCE_INPUT"
echo "Preclinical reference:        $PRECLINICAL_REFERENCE_INPUT"
echo "New preclinical input:        $PRECLINICAL_INPUT"
echo
echo "Preclinical output:           $PRECLINICAL_OUTPUT"
echo "Stats output:                 $STATS_OUTPUT"
echo "===================="

START_TIME=$(date +%s)

python "$SCRIPT_PATH" \
    --clinical_reference_input "$CLINICAL_REFERENCE_INPUT" \
    --preclinical_reference_input "$PRECLINICAL_REFERENCE_INPUT" \
    --preclinical_input "$PRECLINICAL_INPUT" \
    --preclinical_output "$PRECLINICAL_OUTPUT" \
    --ontology_path "$ONTOLOGY_PATH" \
    --root_id "$ROOT_ID" \
    --id_column "$ID_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --min_depth "$MIN_DEPTH" \
    --max_desc "$MAX_DESC" \
    --stats_output "$STATS_OUTPUT"

STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ "$STATUS" -ne 0 ]; then
    echo "MONDO parent mapping failed with exit code $STATUS"
    exit "$STATUS"
fi

echo "Finished assigning MONDO dataset parents in ${DURATION} seconds"