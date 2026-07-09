#!/usr/bin/env bash
#SBATCH --job-name=<JOB_NAME> 
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION> 
#SBATCH --time=47:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-%x-%j.out 
#SBATCH --error=slurm-%x-%j.err
#SBATCH --chdir=. 

set -euo pipefail

module purge
module load <PYTHON_CONDA_MODULE>
eval "$(conda shell.bash hook)"

conda activate multibbq

# Combine all models into a single array
models=(


    "openai/gpt-5"
    "openai/gpt-5-mini"
    "openai/gpt-5-nano"
    "openai/gpt-4o"
    "google/gemini-2.5-pro"
    "google/gemini-2.5-flash"
    "google/gemini-2.5-flash-lite"

)

# Use a single loop to iterate through all models
for model in "${models[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODEL(REALWORLD): $model"
    echo "----------------------------------------------------"
    multibbq run --experiment realworld "$model" --textual_context true --ambiguous true --negative true
    multibbq run --experiment realworld "$model" --textual_context true --ambiguous true --negative false
    multibbq run --experiment realworld "$model" --textual_context true --ambiguous false --negative true
    multibbq run --experiment realworld "$model" --textual_context true --ambiguous false --negative false
done