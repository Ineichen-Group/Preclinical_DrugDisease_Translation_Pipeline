#!/bin/bash
#SBATCH --job-name=preclin_fulltext_cadmus
#SBATCH --time=20:30:00                 # Set a time limit for each job
#SBATCH --output=cadmus_output_%A.log  # Save stdout with job and task ID
#SBATCH --error=cadmus_error_%A.log    # Save stderr with job and task ID
#SBATCH --mem=16G                      # Memory per node

# Add edirect directory to PATH
export PATH=${PATH}:/data/sdonev/cadmus/output/medline/edirect

# Start timer
start_time=$(date +%s)

echo "Running fetch_cadmus_fulltext.py..."
python fetch_cadmus_fulltext.py

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "fetch_cadmus_fulltext.py failed to execute."
    exit 1
else
    echo "fetch_cadmus_fulltext.py ran successfully."
fi

# End timer
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Convert to minutes and seconds
mins=$(( elapsed / 60 ))
secs=$(( elapsed % 60 ))

echo "Time elapsed: ${mins} minute(s) and ${secs} second(s)."
