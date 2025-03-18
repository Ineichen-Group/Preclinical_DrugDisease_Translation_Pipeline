#!/bin/bash
#SBATCH --job-name=study_type_teller
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --mem-per-gpu=32G   # Request 32 GB of memory per GPU
#SBATCH --output=training_output_%j.log  # Save stdout to file
#SBATCH --error=training_error_%j.log    # Save stderr to file

# Parameters
MODEL_SCRIPT="finetune.py"

# Capture the start time
start_time=$(date +%s)

python $MODEL_SCRIPT 

# Capture the end time
end_time=$(date +%s)

# Calculate the duration in minutes
duration=$(( (end_time - start_time) / 60 ))

# Inform the user that the training process has finished and display the elapsed time
echo "Training complete. Duration: $duration minutes."
