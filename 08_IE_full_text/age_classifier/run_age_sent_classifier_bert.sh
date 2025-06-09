#!/bin/bash
#SBATCH --job-name=preclin_ner_inference
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=16G
#SBATCH --output=preclin_ner_age_inference.out
#SBATCH --error=preclin_ner_age_inference.err

# Single-step inference using age_sent_clssifier_bert.py

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"


# -----------------------------
# 1) Configuration
# -----------------------------
# Path to the directory (or HF ID) containing your fine-tuned model:
MODEL_DIR="/shares/animalwelfare.crs.uzh/age_sentence_classifier_final/microsoft_BiomedNLP-BiomedBERT-large-uncased-abstract/best"

# Input CSV must have columns: PMID, sentence_id, sent_txt
INPUT_CSV="./data/regex_age_sentences.csv"

# Output CSV will be created with an additional "predicted_label" column
OUTPUT_CSV="./predictions/sentences_with_bert_age_predictions.csv"

# Path to your inference script (Python file):
PREDICT_SCRIPT="./age_sent_clssifier_bert.py"

# -----------------------------
# 2) Check files/directories
# -----------------------------
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: model directory '$MODEL_DIR' not found." >&2
  exit 1
fi

if [ ! -f "$INPUT_CSV" ]; then
  echo "Error: input CSV '$INPUT_CSV' not found." >&2
  exit 1
fi

if [ ! -f "$PREDICT_SCRIPT" ]; then
  echo "Error: inference script '$PREDICT_SCRIPT' not found." >&2
  exit 1
fi

# -----------------------------
# 3) Run inference
# -----------------------------
echo "============================================"
echo "Starting inference with age_sent_clssifier_bert.py"
echo "  Model Dir:   $MODEL_DIR"
echo "  Input CSV:   $INPUT_CSV"
echo "  Output CSV:  $OUTPUT_CSV"
echo "============================================"
start_time=$(date +%s)

python3 "$PREDICT_SCRIPT" \
  "$MODEL_DIR" \
  "$INPUT_CSV" \
  "$OUTPUT_CSV"

EXIT_CODE=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

if [ $EXIT_CODE -ne 0 ]; then
  echo ">>> ERROR during inference. Exit code: $EXIT_CODE" >&2
  exit $EXIT_CODE
else
  echo "Finished inference in ${minutes}m ${seconds}s"
  echo "Output written to: $OUTPUT_CSV"
fi

echo "All done."
