#!/usr/bin/bash -l
#SBATCH --job-name=fda_metadata
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/fda_metadata_%j.out
#SBATCH --error=logs/fda_metadata_%j.err


# --- Paths and parameters ---
SCRIPT_PATH="fetch_fda_drug_details.py"

OUTPUT_PATH="out/fda_drug_metadata_progress.csv"
TERMS_PATH="out/unique_drug_terms_218510.csv"
MIN_ARTICLES=2
CHECKPOINT_EVERY=1000
TERMS_COL="drug_term_umls_norm_manual_clean"
ARTICLES_COL="n_articles"

# --- Create directories if needed ---
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p logs

echo "[INFO] Starting FDA metadata job on $(hostname)"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Using up to $SLURM_CPUS_PER_TASK CPU cores"
echo "[INFO] Output: $OUTPUT_PATH"
echo "------------------------------------------------------------"

# --- Timing ---
start_time=$(date +%s)

# --- Run Python script ---
python3 "$SCRIPT_PATH" \
  --output "$OUTPUT_PATH" \
  --terms "$TERMS_PATH" \
  --min-articles "$MIN_ARTICLES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --terms-col "$TERMS_COL" \
  --articles-col "$ARTICLES_COL"

# --- Compute total runtime ---
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

echo "------------------------------------------------------------"
echo "[DONE] Job completed in ${hours}h ${minutes}m ${seconds}s"
echo "[INFO] Finished at $(date)"