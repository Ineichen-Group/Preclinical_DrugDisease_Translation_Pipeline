#!/bin/bash

#SBATCH --job-name=mondo_group_clean
#SBATCH --output=logs/mondo_group_clean_%j.out
#SBATCH --error=logs/mondo_group_clean_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

SCRIPT="mondo_clean_names.py"

PRECLINICAL_IN="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/disease_mapped_preclinical_enriched_all_update_2025.csv"

PRECLINICAL_OUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_mondo_cleaned_update_2025.csv"

mkdir -p logs "$(dirname "$PRECLINICAL_OUT")"

echo "===== MONDO GROUP & CLEAN ====="
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "Preclinical in:  $PRECLINICAL_IN"
echo "Preclinical out: $PRECLINICAL_OUT"
echo "==============================="

START=$(date +%s)

python "$SCRIPT" \
    --preclinical_input "$PRECLINICAL_IN" \
    --preclinical_output "$PRECLINICAL_OUT" \
    --preclinical_key PMID \
    --raw_col disease_mondo_term_norm \
    --id_col disease_mondo_termid \
    --grouped_col disease_term_mondo_clean \
    --out_id_col disease_termid_mondo_clean \
    --verbose

STATUS=$?

END=$(date +%s)

if [ "$STATUS" -ne 0 ]; then
    echo "mondo_clean_names.py failed with exit code $STATUS"
    exit "$STATUS"
fi

echo "Finished in $((END - START)) seconds"