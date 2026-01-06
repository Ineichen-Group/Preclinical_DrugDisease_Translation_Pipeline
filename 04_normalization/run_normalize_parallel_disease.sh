#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/job_%x_%j_%a_disease.out
#SBATCH --error=logs/job_%x_%j_%a_disease.err
#SBATCH --array=1-11

ENTITY_TYPE="disease"
COL_TO_MAP="unique_conditions_linkbert_predictions"

DATA_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

# task 1 = clinical; tasks 2-11 = preclinical chunks 1-10
if [ "$TASK_ID" -eq 1 ]; then
    DATASET="clinical"
    CHUNK_ID="all"

    INPUT_FILE="${DATA_DIR}clinical/clinical_raw_annotations.csv"
    OUTPUT_FILE="${DATA_DIR}mapped_to_embeddings_ontologies/clinical/${ENTITY_TYPE}_mapped_clinical_${ENTITY_TYPE}_enriched.csv"
    LINKING_STATS_DIR="nen_stats/clinical_disease/"
else
    DATASET="preclinical"
    CHUNK_ID=$((TASK_ID - 1))
    INPUT_FILE="${DATA_DIR}raw_ner/chunks/ner_chunk_${CHUNK_ID}.csv"
    OUTPUT_FILE="${DATA_DIR}mapped_to_embeddings_ontologies/preclinical_chunks/${ENTITY_TYPE}_mapped_preclinical_${ENTITY_TYPE}_enriched_${CHUNK_ID}.csv"
    LINKING_STATS_DIR="nen_stats/preclinical_disease/chunk_${CHUNK_ID}/"
fi

mkdir -p logs timing_logs "$LINKING_STATS_DIR" "$(dirname "$OUTPUT_FILE")"

# --- set terminology + threshold ---
# (for disease this will always choose mondo, but keeping your structure)
if [ "$ENTITY_TYPE" = "disease" ]; then
    TERMINOLOGY="mondo"
    DIST_THRESHOLD=9.65
else
    TERMINOLOGY="umls"
    DIST_THRESHOLD=8.20
fi

echo "===== DEBUG LOG BEGIN ====="
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "DATASET: $DATASET"
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

echo "Starting normalization for $ENTITY_TYPE (${DATASET}, chunk $CHUNK_ID)"
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

echo "$ENTITY_TYPE,${DATASET},chunk_${CHUNK_ID},${DURATION}" >> timing_logs/${ENTITY_TYPE}_timing.csv
echo "Finished ${DATASET} chunk $CHUNK_ID for $ENTITY_TYPE in ${DURATION} seconds"