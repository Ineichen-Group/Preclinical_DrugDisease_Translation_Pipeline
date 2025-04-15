#!/bin/bash
#SBATCH --job-name=preclin_ner_10_fold_cv
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --array=0-9
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

# Get fold index from SLURM array
fold=$SLURM_ARRAY_TASK_ID

# Define models to experiment with
models=(
    "michiyasunaga/BioLinkBERT-large"
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    "sultan/BioM-ALBERT-xxlarge-PMC"
)

# Define percentages (in case training with a subset of the data wanted)
percentages=(100)

# Loop over models and percentages
for model in "${models[@]}"; do
    for percentage in "${percentages[@]}"; do
        model_name=$(basename "$model")  # e.g., "BioLinkBERT-base"

        echo "Running fold=$fold | model=$model_name | pct=$percentage"
        start_time=$(date +%s)

        python train_bert_ner.py \
            --output_path "./out_strain/${model_name}/fold_${fold}" \
            --model_name_or_path "$model" \
            --train_data_path "./data/k_fold_splits/strain_train_fold_${fold}.json" \
            --val_data_path "./data/k_fold_splits/strain_test_fold_${fold}.json" \
            --test_data_path "./data/k_fold_splits/strain_test_fold_${fold}.json" \
            --n_epochs 15 \
            --percentage "$percentage" \
            --i "$fold"

        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Finished fold=$fold | model=$model_name in $elapsed seconds (~$((elapsed / 60)) minutes)"
    done
done
