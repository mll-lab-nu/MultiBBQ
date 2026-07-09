#!/bin/bash

export HUGGINGFACE_HUB_TOKEN
MODELS_TO_DOWNLOAD=(
  # "Salesforce/blip2-opt-2.7b"
  # "Salesforce/blip2-opt-6.7b"

  # "deepseek-ai/deepseek-vl-1.3b-chat"
  # "deepseek-ai/deepseek-vl-7b-chat"

  # "adept/fuyu-8b"

  # "llava-hf/llava-v1.6-mistral-7b-hf"
  # "llava-hf/llava-v1.6-vicuna-13b-hf"
  # "llava-hf/llava-v1.6-34b-hf"

  # "deepseek-ai/deepseek-vl-1.3b-chat"
  # "deepseek-ai/deepseek-vl-7b-chat"

  # "OpenGVLab/InternVL3_5-1B"
  # "OpenGVLab/InternVL3_5-2B"
  # "OpenGVLab/InternVL3_5-4B"
  # "OpenGVLab/InternVL3_5-8B"
  # "OpenGVLab/InternVL3_5-14B"
  # "OpenGVLab/InternVL3_5-38B"

  # "google/gemma-3-4b-it"
  # "google/gemma-3-1b-it"
  # "google/gemma-3-12b-it"
  # "google/gemma-3-27b-it"


  # "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
  # "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
  # "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

  # "internlm/Intern-S1-mini"
  # "openbmb/MiniCPM-V-4_5"

)


for model in "${MODELS_TO_DOWNLOAD[@]}"
do
  echo "================================================="
  echo "Preparing to download model: ${model}"
  echo "================================================="
  huggingface-cli download \
    --resume-download \
    --local-dir-use-symlinks True \
    "${model}"
done

echo "All specified models have been downloaded."