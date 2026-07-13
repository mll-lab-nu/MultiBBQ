#!/bin/bash
#
# Pre-fetch the open-source checkpoints evaluated in the paper into the local
# HuggingFace cache (set HF_HOME to relocate it). Useful on clusters where the
# compute nodes have no internet access: run this on a login node, then export
# HF_HUB_OFFLINE=1 for the jobs.
#
# Gated models (if any) need `hf auth login` or HF_TOKEN in the environment.

set -euo pipefail

MODELS_TO_DOWNLOAD=(
  "Salesforce/blip2-opt-2.7b"
  "Salesforce/blip2-opt-6.7b"

  "deepseek-ai/deepseek-vl-1.3b-chat"
  "deepseek-ai/deepseek-vl-7b-chat"

  "adept/fuyu-8b"

  "llava-hf/llava-v1.6-mistral-7b-hf"
  "llava-hf/llava-v1.6-vicuna-13b-hf"
  "llava-hf/llava-v1.6-34b-hf"

  "OpenGVLab/InternVL3_5-1B"
  "OpenGVLab/InternVL3_5-2B"
  "OpenGVLab/InternVL3_5-4B"
  "OpenGVLab/InternVL3_5-8B"
  "OpenGVLab/InternVL3_5-14B"
  "OpenGVLab/InternVL3_5-38B"

  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"

  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"

  "openbmb/MiniCPM-V-4_5"

  # AWQ checkpoints, only needed for the quant experiment:
  # "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
  # "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
  # "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
)

for model in "${MODELS_TO_DOWNLOAD[@]}"
do
  echo "================================================="
  echo "Downloading model: ${model}"
  echo "================================================="
  hf download "${model}"
done

echo "All specified models have been downloaded."
