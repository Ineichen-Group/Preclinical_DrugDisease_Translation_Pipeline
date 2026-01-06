#!/bin/bash
#SBATCH --job-name=mondo_group_clean
#SBATCH --output=logs/mondo_group_clean_%j.out
#SBATCH --error=logs/mondo_group_clean_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

SCRIPT="mondo_clean_names.py"

CLINICAL_IN="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/clinical/disease_mapped_clinical_disease_enriched.csv"
PRECLINICAL_IN="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_to_embeddings_ontologies/disease_mapped_preclinical_enriched_all.csv"

CLINICAL_OUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_clinical_data_mondo_cleaned.csv"
PRECLINICAL_OUT="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/mapped_preclinical_data_mondo_cleaned.csv"

mkdir -p logs "$(dirname "$CLINICAL_OUT")" "$(dirname "$PRECLINICAL_OUT")"

echo "===== MONDO GROUP & CLEAN ====="
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Clinical in:   $CLINICAL_IN"
echo "Preclinical in:$PRECLINICAL_IN"
echo "Clinical out:  $CLINICAL_OUT"
echo "Preclinical out:$PRECLINICAL_OUT"
echo "==============================="

START=$(date +%s)

python "$SCRIPT" \
  --clinical_input "$CLINICAL_IN" \
  --preclinical_input "$PRECLINICAL_IN" \
  --clinical_output "$CLINICAL_OUT" \
  --preclinical_output "$PRECLINICAL_OUT" \
  --clinical_key nct_id \
  --preclinical_key PMID \
  --raw_col disease_mondo_term_norm \
  --id_col disease_mondo_termid \
  --grouped_col disease_term_mondo_clean \
  --out_id_col disease_termid_mondo_clean \
  --verbose

END=$(date +%s)
echo "Finished in $((END - START)) seconds"