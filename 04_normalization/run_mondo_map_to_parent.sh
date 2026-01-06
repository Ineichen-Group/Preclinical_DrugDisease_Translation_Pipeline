#!/bin/bash
#SBATCH --job-name=mondo_map_parent
#SBATCH --output=logs/mondo_map_parent.out
#SBATCH --error=logs/mondo_map_parent.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---- paths ----
SCRIPT_PATH="mondo_map_to_parent.py"

# ---- inputs ----
CLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/clinical/disease_mapped_clinical_disease_enriched.csv"
PRECLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/disease_mapped_preclinical_enriched_all.csv"

# ---- outputs ----
CLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_data_with_mondo_parents.csv"
PRECLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_with_mondo_parents.csv"

# ---- ontology + params ----
ONTOLOGY_PATH="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mondo/mondo.owl"
ROOT_ID="MONDO:0000001"

ID_COLUMN="disease_mondo_termid"
LABEL_COLUMN="disease_mondo_term_norm"

MIN_DEPTH=5
MAX_DESC=500

mkdir -p logs "$(dirname "$CLINICAL_OUTPUT")" "$(dirname "$PRECLINICAL_OUTPUT")"

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Input Clinical: $CLINICAL_INPUT"
echo "Input Preclinical: $PRECLINICAL_INPUT"
echo "Output Clinical: $CLINICAL_OUTPUT"
echo "Output Preclinical: $PRECLINICAL_OUTPUT"
echo "===================="

START_TIME=$(date +%s)

python "$SCRIPT_PATH" \
  --clinical_input "$CLINICAL_INPUT" \
  --preclinical_input "$PRECLINICAL_INPUT" \
  --clinical_output "$CLINICAL_OUTPUT" \
  --preclinical_output "$PRECLINICAL_OUTPUT" \
  --ontology_path "$ONTOLOGY_PATH" \
  --root_id "$ROOT_ID" \
  --id_column "$ID_COLUMN" \
  --label_column "$LABEL_COLUMN" \
  --min_depth "$MIN_DEPTH" \
  --max_desc "$MAX_DESC"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Finished assigning MONDO dataset parents in ${DURATION} seconds"