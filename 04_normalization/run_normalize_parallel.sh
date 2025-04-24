#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/job_%x_%j_%a.out
#SBATCH --error=logs/job_%x_%j_%a.err
#SBATCH --array=1-10

ENTITY_TYPE="disease"
COL_TO_MAP="linkbert_aact_mapped_conditions"

CHUNK_ID=$SLURM_ARRAY_TASK_ID
DATA_DIR="data/"
INPUT_FILE="mapped_to_dict/clinical/clinical_combined_annotations.csv" #chunks/dict_mapped_ner_chunk_${CHUNK_ID}.csv"
OUTPUT_FILE="mapped_to_embeddings_ontologies/clinical/${ENTITY_TYPE}_mapped_clinical.csv" #${ENTITY_TYPE}_mapped_ner_chunk_${CHUNK_ID}.csv"
LINKING_STATS_DIR="nen_stats/${CHUNK_ID}_clinical_"

# Timing start
START_TIME=$(date +%s)

echo "Starting normalization for $ENTITY_TYPE (chunk $CHUNK_ID)"
python neural_based_nen.py --type "$ENTITY_TYPE" --col_to_map "$COL_TO_MAP" --data_dir "$DATA_DIR" --input "$INPUT_FILE" --output "$OUTPUT_FILE" --stats_dir "$LINKING_STATS_DIR"

# Timing end
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Write to timing log
echo "$ENTITY_TYPE,chunk_${CHUNK_ID},${DURATION}" >> timing_logs/${ENTITY_TYPE}_timing.csv

echo "Finished chunk $CHUNK_ID for $ENTITY_TYPE in ${DURATION} seconds"
