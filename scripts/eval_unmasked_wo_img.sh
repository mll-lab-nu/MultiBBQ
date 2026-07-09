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


# Combine all models into a single array
models=(
"llava-hf/llava-v1.6-mistral-7b-hf"
"Qwen/Qwen2.5-VL-3B-Instruct"
"OpenGVLab/InternVL3_5-4B"
"google/gemma-3-4b-it"

"llava-hf/llava-v1.6-vicuna-13b-hf"
"llava-hf/llava-v1.6-34b-hf"
"deepseek-ai/deepseek-vl-1.3b-chat"
"deepseek-ai/deepseek-vl-7b-chat"
"adept/fuyu-8b"
"Salesforce/blip2-opt-2.7b"
"Salesforce/blip2-opt-6.7b"
"Qwen/Qwen2.5-VL-7B-Instruct"

"Qwen/Qwen2.5-VL-72B-Instruct"

"OpenGVLab/InternVL3_5-1B"
"OpenGVLab/InternVL3_5-2B"
"OpenGVLab/InternVL3_5-8B"
"OpenGVLab/InternVL3_5-14B"
"OpenGVLab/InternVL3_5-38B"
"openbmb/MiniCPM-V-4_5"
"google/gemma-3-12b-it"
"google/gemma-3-27b-it"

)

# Use a single loop to iterate through all models
for model in "${models[@]}"; do
    echo "----------------------------------------------------"
    echo "EVALUATING MODEL(UNMASKED WITHOUT IMAGE): $model"
    echo "----------------------------------------------------"
    multibbq run --experiment unmasked_wo_img "$model" --textual_context true --ambiguous true --negative true
    multibbq run --experiment unmasked_wo_img "$model" --textual_context true --ambiguous true --negative false
    multibbq run --experiment unmasked_wo_img "$model" --textual_context true --ambiguous false --negative true
    multibbq run --experiment unmasked_wo_img "$model" --textual_context true --ambiguous false --negative false
done
