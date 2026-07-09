---
license: cc-by-4.0
pretty_name: MultiBBQ experimental outputs
task_categories:
- visual-question-answering
language:
- en
tags:
- fairness
- social-bias
- multimodal
- vision-language
- benchmark-results
---

# MultiBBQ: experimental outputs

Raw model outputs and computed metrics for the paper *Fairness Failure Modes of Multimodal
LLMs*. These are reproduction artifacts: the exact predictions behind the paper's tables plus
the Fairness / Bias / Unknown-rate numbers derived from them. This is not a dataset to train
on.

- **Paper:** *Fairness Failure Modes of Multimodal LLMs*
- **Code:** https://github.com/mll-lab-nu/MultiBBQ
- **Core dataset:** https://huggingface.co/datasets/MLL-Lab/MultiBBQ
- **Perturbations:** https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations
- **License:** CC-BY-4.0

## Layout

```
MLL-Lab/MultiBBQ-results
├── results/     # raw inference outputs, one directory per experiment
│   └── <experiment>/<org>/<model>/<file>.json
└── analysis/    # computed metrics + aggregated CSVs, mirroring results/
    └── <experiment>/...            (+ combined_metrics.json, CSV summaries)
```

Everything in `analysis/` is regenerable from `results/` with the code, so `results/` is the
source of truth.

## Experiments (directory names)

Each experiment is a directory under `results/`. The prefix is the image generator
(`gpt_image_gen` = GPT-Image-1, `imagen4ultra_image_gen` = Imagen 4 Ultra).

| Directory | Experiment |
|---|---|
| `gpt_image_gen_main`, `imagen4ultra_image_gen_main` | main run (visual-only + visual-language, ambiguous + disambiguated) for both generators |
| `gpt_image_gen_reasoning`, `gpt_image_gen_nonreasoning_w_fairness`, `gpt_image_gen_reasoning_w_fairness` | reasoning vs non-reasoning mode |
| `gpt_image_gen_temp_0.2` … `gpt_image_gen_temp_1.0` | decoding-temperature sweep |
| `gpt_image_gen_quant` | quantized inference |
| `gpt_image_gen_realworld`, `gpt_image_gen_main4realworld` | real face images (transferability) |
| `gpt_image_gen_unmasked_w_img`, `gpt_image_gen_unmasked_wo_img` | unmasked BBQ text, with and without the image (text-context / LLM ablation) |
| `gpt_image_gen_brightness_up/down`, `gpt_image_gen_contrast_up/down`, `gpt_image_gen_compression`, `gpt_image_gen_noise`, `gpt_image_gen_resize_l/s` | image-perturbation robustness (see [MultiBBQ-perturbations](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations)) |

## Models

Outputs cover the 28 models across 11 families reported in the paper (checkpoints in
`results/*_main/`):

- **GPT-4o:** gpt-4o
- **GPT-5:** gpt-5, gpt-5-mini, gpt-5-nano
- **Google Gemini:** gemini-2.5-flash, gemini-2.5-flash-lite
- **Google Gemma:** gemma-3-4b/12b/27b-it
- **Qwen:** Qwen2.5-VL-3B/7B/32B/72B-Instruct
- **InternVL (OpenGVLab):** InternVL3_5-1B/2B/4B/8B/14B/38B
- **LLaVA-NeXT (llava-hf):** llava-v1.6-mistral-7b, -vicuna-13b, -34b
- **DeepSeek-VL:** deepseek-vl-1.3b-chat, -7b-chat
- **MiniCPM-V:** MiniCPM-V-4_5
- **BLIP-2 (Salesforce):** blip2-opt-2.7b, -6.7b
- **Fuyu (adept):** fuyu-8b

## File format

Each `results/` file is one (model, modality, question-framing, context) slice. The filename
encodes the setting, for example:

```
Qwen2.5_72B_visual_language_nonnegative_disambiguous.json
                └ modality ┘└ framing ┘└ context ┘
```

The JSON has a top-level `data` list; each record is one example:

| Field | Description |
|---|---|
| `image` | image path used at inference |
| `category` | race / gender / religion / age |
| `options` | the answer options shown |
| `pred` | the model's raw prediction |
| `correct_option_idx` | gold answer index |
| `stereotype_group_idx`, `nonstereotype_group_idx` | option indices of the two subgroups |
| `unk_label_idx` | option index of *Unknown* |

## Reproduce the metrics

```bash
pip install -e .        # from the MultiBBQ code repo
# score one experiment: raw outputs -> Fairness / Bias / Unknown-rate + CSV summaries
multibbq pipeline --input results/gpt_image_gen_main --output analysis/gpt_image_gen_main
```

The metric subcommands run in a light environment (only pandas), no GPU needed.

## Citation

```bibtex
@article{chen2026multibbq,
  title   = {Fairness Failure Modes of Multimodal LLMs},
  author  = {Chen, Canyu and Cai, Anglin and Nwatu, Joan and Li, Yale and
             Hullman, Jessica and Mihalcea, Rada and McKeown, Kathleen and Li, Manling},
  year    = {2026},
  note    = {MultiBBQ. Project: https://multibbq.github.io},
}
```
