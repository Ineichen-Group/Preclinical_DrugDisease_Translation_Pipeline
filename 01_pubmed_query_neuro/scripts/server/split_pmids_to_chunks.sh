#!/bin/bash
#SBATCH --job-name=split_file_job     # Job name
#SBATCH --output=split_file_output.txt # Output log file
#SBATCH --error=split_file_error.txt   # Error log file
#SBATCH --time=00:10:00               # Maximum runtime (hh:mm:ss)
#SBATCH --mem=1G                      # Memory required
#SBATCH --cpus-per-task=1             # Number of CPUs required

# Parse input parameters
input_file=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input-file)
            input_file="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 --input-file <input_file>"
            exit 1
            ;;
    esac
    shift
done

# Check if the input file is provided
if [ -z "$input_file" ]; then
    echo "Error: --input-file parameter is required"
    echo "Usage: $0 --input-file <input_file>"
    exit 1
fi

# Verify that the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found!"
    exit 1
fi

# Lines per chunk
lines_per_chunk=5000

# Total lines in the file
total_lines=$(wc -l < "$input_file")

# Calculate the number of chunks needed
num_chunks=$(( (total_lines + lines_per_chunk - 1) / lines_per_chunk ))

# Loop to split the file manually
for ((i=0; i<num_chunks; i++)); do
    start_line=$(( i * lines_per_chunk + 1 ))
    end_line=$(( start_line + lines_per_chunk - 1 ))
    chunk_file=$(printf "pmid_chunk_%03d.txt" "$i")
    
    # Extract the lines for this chunk
    sed -n "${start_line},${end_line}p" "$input_file" > "$chunk_file"
done

echo "File split into $num_chunks chunks."
