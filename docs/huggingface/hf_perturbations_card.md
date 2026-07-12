---
license: cc-by-4.0
pretty_name: MultiBBQ perturbations
task_categories:
- visual-question-answering
language:
- en
tags:
- fairness
- social-bias
- multimodal
- vision-language
- robustness
- image-perturbation
size_categories:
- 1K<n<10K
---

<br>

<p align="center">
  <img src="https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations/resolve/main/logo_horizontal.png" alt="MultiBBQ logo" width="620"/>
</p>


<h1 align="center">MultiBBQ: image perturbations</h1>


<p align="center">
  <a href="https://multibbq.github.io"><img src="https://img.shields.io/badge/🏠_Project-4285F4?style=for-the-badge&logoColor=white" alt="Project page"></a>
  <a href="https://multibbq.github.io"><img src="https://img.shields.io/badge/📄_Paper-DC143C?style=for-the-badge&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/MLL-Lab/MultiBBQ"><img src="https://img.shields.io/badge/🤗_Dataset-FFD21E?style=for-the-badge&logoColor=black" alt="HuggingFace dataset"></a>
  <a href="https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results"><img src="https://img.shields.io/badge/📊_Results-FFD21E?style=for-the-badge&logoColor=black" alt="HuggingFace results"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/⚖️_Code-MIT-4285F4?style=for-the-badge&logoColor=white" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://drive.google.com/file/d/1OZcaRvlcB6uqkRgm5ve-ds0xS4TuW_6Z/view?usp=sharing"><img src="https://img.shields.io/badge/🏆_Best_Paper_Award-ACL_2026_TrustNLP_Workshop-FFB300?style=for-the-badge&labelColor=8B6914&logoColor=white" alt="Best Paper Award - ACL 2026 Workshop on Trustworthy NLP"></a>
</p>


Image-level perturbation sets used for the **robustness** experiments in *Fairness Failure
Modes of Multimodal LLMs*. Each set is the GPT-Image-1 image collection from
[MLL-Lab/MultiBBQ](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) with a single, controlled
transform applied. Evaluating on a perturbed set measures how stable a model's fairness
behavior is under everyday image degradations.

- **Paper:** *Fairness Failure Modes of Multimodal LLMs*
- **Code:** https://github.com/mll-lab-nu/MultiBBQ
- **Core dataset:** https://huggingface.co/datasets/MLL-Lab/MultiBBQ
- **Results:** https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results
- **License:** CC-BY-4.0

Perturbations are applied to the **GPT-Image-1** images only. The text metadata is unchanged;
use the metadata from the core dataset. These are raw image trees (not a `load_dataset`
config), because they reuse the core metadata and only swap the pixels.

## Sets

Eleven perturbation sets, each mirroring the core image layout (`textual/` = visual-language
images, `visual/` = visual-only images).

| Folder | Perturbation |
|---|---|
| `gpt_image_gen_brightness`, `gpt_image_gen_brightness_up`, `gpt_image_gen_brightness_down` | brightness shift (baseline / brighter / darker) |
| `gpt_image_gen_contrast`, `gpt_image_gen_contrast_up`, `gpt_image_gen_contrast_down` | contrast change (baseline / higher / lower) |
| `gpt_image_gen_compression` | JPEG compression artifacts |
| `gpt_image_gen_noise` | additive noise |
| `gpt_image_gen_resize_l`, `gpt_image_gen_resize_s` | resize larger / smaller |
| `gpt_image_gen_label` | on-image text label overlay |

```
MLL-Lab/MultiBBQ-perturbations
└── gpt_image_gen_<perturbation>/
    ├── textual/   # visual-language images
    └── visual/    # visual-only images
```

Each set holds the same images as the core GPT-Image-1 collection (818 files per set; the
`label` set has 816), so paths line up one-to-one with the core `image_path` values after
substituting the folder name.

## Use it with the toolkit

The harness fetches these sets and runs the perturbation experiments for you:

```bash
pip install "multibbq[hf]"
multibbq download                       # includes the perturbation trees
# augmented-image robustness (aug_img) with a chosen perturbation
multibbq run "OpenGVLab/InternVL3_5-8B" --experiment aug_img --img_aug_type noise
# on-image label overlay (img_label)
multibbq run "OpenGVLab/InternVL3_5-8B" --experiment img_label
```

Valid `--img_aug_type` values: `brightness`, `brightness_up`, `brightness_down`, `contrast`,
`contrast_up`, `contrast_down`, `compression`, `noise`, `resize_l`, `resize_s`.

The corresponding model outputs and computed metrics are in
[MLL-Lab/MultiBBQ-results](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results)
(directories named `gpt_image_gen_<perturbation>`).

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