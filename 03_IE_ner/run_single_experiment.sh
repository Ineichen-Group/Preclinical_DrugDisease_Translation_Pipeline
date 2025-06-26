#!/bin/bash
#SBATCH --job-name=preclin_ner_10_fold_cv
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=32G  # Request 32 GB of memory per GPU

# Define the list of percentage values and i values
percentages=(100)
i_values=(1)

# Loop over percentage and i values
for percentage in "${percentages[@]}"; do
    for i in "${i_values[@]}"; do
        echo "Running experiment with percentage=$percentage and i=$i"
        python train_bert_ner.py \
            --output_path "./out_full_ds" \
            --model_name_or_path "michiyasunaga/BioLinkBERT-base" \
            --train_data_path "./data/full_ds_drug_disease_ner.json" \
            --test_data_path "./data/test_fold_0.json" \
            --n_epochs 10 --percentage "$percentage" --i "$i"
    done
done
