#!/bin/bash
#SBATCH --job-name=embed_ont
#SBATCH --output=embed_ont.out
#SBATCH --error=embed_ont.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# run script
python embed_ontology.py
