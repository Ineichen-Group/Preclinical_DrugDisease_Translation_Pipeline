#!/bin/bash
#SBATCH --job-name=preclin_studytypeteller
#SBATCH --array=0-20%20  # Process only 20 jobs at a time
#SBATCH --gpus=1                        # Request 1 GPU per task
#SBATCH --time=01:30:00                 # Set a time limit for each job
#SBATCH --mem-per-gpu=32G               # Request 32 GB of memory per GPU
#SBATCH --output=inference_logs/inference_output_%A_%a.log  # Save stdout with job and task ID
#SBATCH --error=inference_logs/inference_error_%A_%a.log    # Save stderr with job and task ID

# Parameters
INFERENCE_SCRIPT="./inference_studytypeteller.py"
TUNED_MODEL="./models/transformers/models_binary/Scibert/best_model_Scibert_binary.pt"
MODEL="allenai/scibert_scivocab_uncased"

# Use the Slurm array task ID as the PUBMED_CHUNK_ID
PUBMED_CHUNK_ID=$SLURM_ARRAY_TASK_ID
PUBMED_DATA_PATH="../pubmed_abstracts/pubmed_results_round8_missing"
OUTPUTS_DATA_PATH="./model_predictions/pubmed_results_r8/"

# Define the input file and output file for the current chunk
FILE_FOR_INFERENCE="${PUBMED_DATA_PATH}/pmid_contents_chunk_${PUBMED_CHUNK_ID}.txt"
OUTPUT_FILE="${OUTPUTS_DATA_PATH}/predictions_chunk_${PUBMED_CHUNK_ID}.txt"

# Print the current configuration for debugging
echo "Running BERT model inference on chunk ID ${PUBMED_CHUNK_ID}..."
echo "Model: $MODEL, Script: $INFERENCE_SCRIPT"
echo "Input file: $FILE_FOR_INFERENCE"
echo "Output file: $OUTPUT_FILE"

# Execute the Python script with specified parameters
python $INFERENCE_SCRIPT --model_name_or_path $MODEL \
                         --trained_model_path $TUNED_MODEL \
                         --pubmed_file $FILE_FOR_INFERENCE \
                         --output_file $OUTPUT_FILE

# Inform the user that the process has finished
echo "Inference complete for chunk ID ${PUBMED_CHUNK_ID}."
