#!/bin/bash
#SBATCH --job-name=umls_map_parent
#SBATCH --output=logs/umls_map_parent.out
#SBATCH --error=logs/umls_map_parent.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ---- paths ----
SCRIPT_PATH="umls_map_to_parent.py"

MRREL_PATH="./data/umls/mrrel_all_drug_rela_20251209.csv"
ID_TO_TERM_MAP="./data/umls/umls_id_to_term_map.json"

CLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/clinical/drug_mapped_clinical_drug_enriched.csv"
PRECLINICAL_INPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/drug_mapped_preclinical_enriched_all.csv"

CLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_drug_data_with_umls_parents.csv"
PRECLINICAL_OUTPUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_drug_data_with_umls_parents.csv"

STATS_FOLDER="./data/umls/"

mkdir -p logs "$STATS_FOLDER"

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "===================="

START_TIME=$(date +%s)

python "$SCRIPT_PATH" \
  --mrrel_path "$MRREL_PATH" \
  --id_to_term_map_path "$ID_TO_TERM_MAP" \
  --clinical_input "$CLINICAL_INPUT" \
  --preclinical_input "$PRECLINICAL_INPUT" \
  --clinical_output "$CLINICAL_OUTPUT" \
  --preclinical_output "$PRECLINICAL_OUTPUT" \
  --stats_folder "$STATS_FOLDER"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Finished assigning UMLS parents in ${DURATION} seconds"
