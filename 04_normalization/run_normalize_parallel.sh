#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/job_%x_%j_%a.out
#SBATCH --error=logs/job_%x_%j_%a.err
#SBATCH --array=1-10

ENTITY_TYPE="drug" #"disease"
COL_TO_MAP="linkbert_mapped_drugs" #"linkbert_mapped_conditions"

CHUNK_ID=$SLURM_ARRAY_TASK_ID
DATA_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/"
INPUT_FILE="${DATA_DIR}mapped_to_dict/clinical/clinical_combined_annotations.csv"
OUTPUT_FILE="${DATA_DIR}mapped_to_embeddings_ontologies/clinical/${ENTITY_TYPE}_mapped_clinical_drug_enriched.csv"
LINKING_STATS_DIR="nen_stats/${CHUNK_ID}_clinical_"

# --- set terminology + threshold (can tweak per type) ---
if [ "$ENTITY_TYPE" = "disease" ]; then
    TERMINOLOGY="mondo"
    DIST_THRESHOLD=9.65
else
    TERMINOLOGY="umls"
    DIST_THRESHOLD=8.20
fi

echo "===== DEBUG LOG BEGIN ====="
echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "ENTITY_TYPE: $ENTITY_TYPE"
echo "COL_TO_MAP: $COL_TO_MAP"
echo "DATA_DIR: $DATA_DIR"
echo "INPUT_FILE: $INPUT_FILE"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "LINKING_STATS_DIR: $LINKING_STATS_DIR"
echo "TERMINOLOGY: $TERMINOLOGY"
echo "DIST_THRESHOLD: $DIST_THRESHOLD"
echo "===== DEBUG LOG END ====="

# Timing start
START_TIME=$(date +%s)

echo "Starting normalization for $ENTITY_TYPE (chunk $CHUNK_ID)"
python neural_based_nen.py \
  --type "$ENTITY_TYPE" \
  --col_to_map "$COL_TO_MAP" \
  --data_dir "$DATA_DIR" \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_FILE" \
  --stats_dir "$LINKING_STATS_DIR" \
  --terminology "$TERMINOLOGY" \
  --dist_threshold "$DIST_THRESHOLD"

# Timing end
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "$ENTITY_TYPE,chunk_${CHUNK_ID},${DURATION}" >> timing_logs/${ENTITY_TYPE}_timing.csv

echo "Finished chunk $CHUNK_ID for $ENTITY_TYPE in ${DURATION} seconds"
