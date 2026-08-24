#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/job_%x_%j_%a_drug.out
#SBATCH --error=logs/job_%x_%j_%a_drug.err
#SBATCH --array=1-10

ENTITY_TYPE="drug"
COL_TO_MAP="unique_interventions_linkbert_predictions"

DATA_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/"

CHUNK_ID="${SLURM_ARRAY_TASK_ID}"

#INPUT_FILE="${DATA_DIR}raw_ner/chunks/ner_chunk_${CHUNK_ID}.csv"
INPUT_FILE="${DATA_DIR}raw_ner/chunks/update_2025/ner_chunk_${CHUNK_ID}.csv"
#OUTPUT_FILE="${DATA_DIR}mapped_to_embeddings_ontologies/preclinical_chunks/${ENTITY_TYPE}_mapped_preclinical_${ENTITY_TYPE}_enriched_${CHUNK_ID}.csv"
OUTPUT_FILE="${DATA_DIR}mapped_to_embeddings_ontologies/preclinical_chunks/update_2025/${ENTITY_TYPE}_mapped_preclinical_${ENTITY_TYPE}_${CHUNK_ID}.csv"
LINKING_STATS_DIR="nen_stats/preclinical_chunks/chunk_${CHUNK_ID}/"

mkdir -p \
    logs \
    timing_logs \
    "$LINKING_STATS_DIR" \
    "$(dirname "$OUTPUT_FILE")"

# Set terminology and threshold
TERMINOLOGY="umls"
DIST_THRESHOLD=8.20

echo "===== DEBUG LOG BEGIN ====="
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "CHUNK_ID: $CHUNK_ID"
echo "ENTITY_TYPE: $ENTITY_TYPE"
echo "COL_TO_MAP: $COL_TO_MAP"
echo "DATA_DIR: $DATA_DIR"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "LINKING_STATS_DIR: $LINKING_STATS_DIR"
echo "TERMINOLOGY: $TERMINOLOGY"
echo "DIST_THRESHOLD: $DIST_THRESHOLD"
echo "===== DEBUG LOG END ====="

START_TIME=$(date +%s)

echo "Starting normalization for ${ENTITY_TYPE}, chunk ${CHUNK_ID}"

python neural_based_nen.py \
    --type "$ENTITY_TYPE" \
    --col_to_map "$COL_TO_MAP" \
    --data_dir "$DATA_DIR" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --stats_dir "$LINKING_STATS_DIR" \
    --terminology "$TERMINOLOGY" \
    --dist_threshold "$DIST_THRESHOLD"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "${ENTITY_TYPE},chunk_${CHUNK_ID},${DURATION}" \
    >> "timing_logs/${ENTITY_TYPE}_timing.csv"

echo "Finished chunk ${CHUNK_ID} for ${ENTITY_TYPE} in ${DURATION} seconds"