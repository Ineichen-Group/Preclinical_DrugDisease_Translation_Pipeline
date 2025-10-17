#!/usr/bin/bash -l
#SBATCH --job-name=clean_chunk
#SBATCH --gpus=H100:1          # <-- request a specific GPU type (e.g., L40S, A100, V100)
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=clean_with_llm_job_%A_%a.out
#SBATCH --error=clean_with_llm_job_%A_%a.err
#SBATCH --array=0-0%1            # <- set real range at submit time; %N limits concurrent tasks

# --- CONFIG ---
CHUNKS_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/chunks_for_LLM_cleaning"
OUT_DIR="/shares/animalwelfare.crs.uzh/Preclinical_Pipeline/04_normalization/data/mapped_all/chunks_cleaned_via_LLM"
MODEL_DIR="/shares/animalwelfare.crs.uzh/llms/DeepSeek-R1-Distill-Qwen-32B"

PROMPT_ID="prompt1_32B_FS"
TARGET_COL="conditions"
ENTITY_TYPE="DISEASE"
CHECKPOINT_EVERY=1000
TP_SIZE=1
MAX_LEN=8192

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# (optional) your env
# module load python/3.11
# source ~/data/conda/bin/activate torch-huggingface

mkdir -p "$OUT_DIR"

# Build an indexed list of CSVs (sorted for determinism)
mapfile -t FILES < <(find "$CHUNKS_DIR" -type f -name "*.csv" | sort)

# Bounds check
IDX="${SLURM_ARRAY_TASK_ID}"
TOTAL="${#FILES[@]}"
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Array index $IDX out of range (0..$((TOTAL-1)))"; exit 1
fi

INPUT_CSV="${FILES[$IDX]}"
BASENAME="$(basename "$INPUT_CSV" .csv)"
OUTPUT_CSV="${OUT_DIR}/${BASENAME}_clean.csv"

echo "[JOB $SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] → $INPUT_CSV"
python clean_with_llm.py \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --prompt-id "$PROMPT_ID" \
  --target-col "$TARGET_COL" \
  --entity-type "$ENTITY_TYPE" \
  --model-dir "$MODEL_DIR" \
  --max-len "$MAX_LEN" \
  --tp "$TP_SIZE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --llm-only

echo "[DONE] $OUTPUT_CSV"
