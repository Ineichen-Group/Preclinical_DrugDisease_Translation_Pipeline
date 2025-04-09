#!/bin/bash
#SBATCH --job-name=sapbert_norm
#SBATCH --time=20:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/job_%x_%j_%a.out
#SBATCH --error=logs/job_%x_%j_%a.err
#SBATCH --array=1-10

ENTITY_TYPE=$1

if [[ "$ENTITY_TYPE" != "disease" && "$ENTITY_TYPE" != "drug" ]]; then
    echo "Invalid entity type: $ENTITY_TYPE. Use 'disease' or 'drug'."
    exit 1
fi

CHUNK_ID=$SLURM_ARRAY_TASK_ID
INPUT_FILE="chunks/dict_mapped_ner_chunk_${CHUNK_ID}.csv"

# Timing start
START_TIME=$(date +%s)

echo "Starting normalization for $ENTITY_TYPE (chunk $CHUNK_ID)"
python neural_based_nen.py --type "$ENTITY_TYPE" --input "$INPUT_FILE"

# Timing end
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Write to timing log
echo "$ENTITY_TYPE,chunk_${CHUNK_ID},${DURATION}" >> timing_logs/${ENTITY_TYPE}_timing.csv

echo "Finished chunk $CHUNK_ID for $ENTITY_TYPE in ${DURATION} seconds"
