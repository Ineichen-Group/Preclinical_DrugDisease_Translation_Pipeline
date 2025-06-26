#!/bin/bash
#SBATCH --job-name=inference_jobs       # Job name
#SBATCH --output=logs/%A_%a.out             # %A is the job ID, %a is the array index
#SBATCH --error=logs/%A_%a.err              # Error log for the job
#SBATCH --gres=gpu:1                   # Reserve 1 GPU per job
#SBATCH --array=1-617                   # Job array for 11 tasks (1 to 11)
#SBATCH --time=03:00:00                # Set max runtime for each job
#SBATCH --mem=16GB                     # Memory allocation
#SBATCH --cpus-per-task=4              # Number of CPUs per task

# Construct the test file path using SLURM_ARRAY_TASK_ID
TEST_DATA_CSV="./pubmed_animal_docs_large/pubmed_filtered_animal_for_NER_chunk_${SLURM_ARRAY_TASK_ID}.csv" #pubmed_filtered_animal_5524202_for_NER_chunk_${SLURM_ARRAY_TASK_ID}.csv"

# Model configuration
MODEL_NAME="michiyasunaga/BioLinkBERT-base"
MODEL_PATH="out_full_ds/results/michiyasunaga_BioLinkBERT-base/epochs_10_data_size_100_iter_1/"
OUTPUT_DIR="./model_predictions/"
OUTPUT_FILE_SUFFIX="_part_${SLURM_ARRAY_TASK_ID}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Capture the start time
start_time=$(date +%s)

# Print job details
echo "Running inference job..."
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Test file: $TEST_DATA_CSV"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"

# Run inference
python inference_ner_annotations.py \
    --test_data_csv $TEST_DATA_CSV \
    --output_dir $OUTPUT_DIR \
    --output_file_suffix $OUTPUT_FILE_SUFFIX \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH

# Capture the end time
end_time=$(date +%s)

# Calculate duration
duration=$(( (end_time - start_time) / 60 ))

# Inform the user of completion
echo "Inference completed for $TEST_DATA_CSV. Duration: $duration minutes."
