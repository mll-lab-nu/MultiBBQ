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

models=(
    "openai/gpt-4o"
    # "openai/gpt-5"
    # "openai/gpt-5-mini"
    # "openai/gpt-5-nano"
    # "google/gemini-2.5-pro"
    # "google/gemini-2.5-flash"
    # "google/gemini-2.5-flash-lite"

)


# The generic-label image set covers the visual-only modality only
# (see docs/huggingface/hf_perturbations_card.md), so there are no
# --textual_context true conditions here.
for model in "${models[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODEL: $model"
    echo "----------------------------------------------------"
    multibbq run --experiment img_label "$model" --textual_context false --ambiguous true --negative true --img_aug_type "label"
    multibbq run --experiment img_label "$model" --textual_context false --ambiguous true --negative false --img_aug_type "label"
done