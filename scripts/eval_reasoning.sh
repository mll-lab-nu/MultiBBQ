#!/bin/bash
#
# Minimal, repo-friendly Slurm script template.
# Usage (recommended): run `sbatch` from the project root so relative paths work as expected.

#SBATCH --job-name=<JOB_NAME>               # Appears in squeue/sacct; also used by %x in log filenames
#SBATCH --account=<ACCOUNT>                 # Cluster allocation/account (user must set)
#SBATCH --partition=<GPU_PARTITION>         # GPU partition/queue name (user must set)
#SBATCH --gres=gpu:<GPU_TYPE>:<GPU_COUNT>   # GPU request, e.g., gpu:a100:4 (syntax varies by cluster)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                  # Keep consistent with your workload's CPU needs
#SBATCH --mem=<MEM>                         # Total memory for the job (user must set)
#SBATCH --time=47:30:00                     # Max runtime
#SBATCH --exclusive                         # Exclusive node access (remove if you don't need it)
#SBATCH --output=logs/model_logs/%x-%j.out  # %x=job name, %j=job id (ensure directory exists)
#SBATCH --error=logs/model_logs/%x-%j.err
#SBATCH --chdir=.                           # Run in the submit directory (recommended: project root)

set -euo pipefail

# --- Environment setup (cluster-dependent) ---
# Replace the module name below with your cluster's Python/Conda module, or remove if not applicable.
module purge
module load <PYTHON_CONDA_MODULE>
eval "$(conda shell.bash hook)"
conda activate <CONDA_ENV_NAME>

reasoning_types=(
    'reasoning'
    'nonreasoning_w_fairness'
    'reasoning_w_fairness'
)
# The open-source models of the mitigation study (paper Section: Bias Mitigation);
# the API models (GPT / Gemini) run via eval_reasoning_cpu.sh.
models=(
"adept/fuyu-8b"
"llava-hf/llava-v1.6-mistral-7b-hf"
"google/gemma-3-4b-it"
"google/gemma-3-12b-it"
"google/gemma-3-27b-it"
"openbmb/MiniCPM-V-4_5"
"OpenGVLab/InternVL3_5-1B"
"OpenGVLab/InternVL3_5-2B"
"OpenGVLab/InternVL3_5-4B"
"OpenGVLab/InternVL3_5-8B"
"OpenGVLab/InternVL3_5-14B"
)

for rt in "${reasoning_types[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODE: $rt"
    echo "----------------------------------------------------"
    for model in "${models[@]}"; do
        echo "----------------------------------------------------"
        echo "EVALUATING MODEL(REASONING): $model"
        echo "----------------------------------------------------"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous true --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous true --negative false --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous false --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context true --ambiguous false --negative false --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context false --ambiguous true --negative true --reasoning_mode "$rt"
        multibbq run --experiment reasoning "$model" --textual_context false --ambiguous true --negative false --reasoning_mode "$rt"
    done
done