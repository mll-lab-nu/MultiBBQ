---
license: cc-by-4.0
pretty_name: MultiBBQ real-world images
task_categories:
- visual-question-answering
language:
- en
tags:
- fairness
- social-bias
- multimodal
- vision-language
- real-images
size_categories:
- n<1K
---

<br>

<p align="center">
  <img src="https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld/resolve/main/logo_horizontal.png" alt="MultiBBQ logo" width="620"/>
</p>


<h1 align="center">MultiBBQ: real-world images</h1>


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


Real-photo image set used for the **real-image transferability** experiment in *Fairness
Failure Modes of Multimodal LLMs*. Each image pairs two real photographs side by side,
mirroring the two-person layout of the synthetic
[MLL-Lab/MultiBBQ](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) images. Evaluating on
this set checks that conclusions drawn from the synthetic benchmark carry over to real
photos: in the paper, fairness scores on synthetic and real images are highly consistent
(Pearson *r* = 0.9787).

- **Paper:** *Fairness Failure Modes of Multimodal LLMs*
- **Code:** https://github.com/mll-lab-nu/MultiBBQ
- **Core dataset:** https://huggingface.co/datasets/MLL-Lab/MultiBBQ
- **Companion repos:** [MultiBBQ-perturbations](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) (robustness image sets) · [MultiBBQ-results](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results) (model outputs + metrics)
- **License:** CC-BY-4.0

This is a raw image tree (not a `load_dataset` config): the harness reuses the text
metadata of the core dataset and matches images to questions by file name, so only the
pixels live here.

## Layout

```
MLL-Lab/MultiBBQ-realworld
└── real_world_image/
    └── {modality}_{category}_q{q_id}_c{c_id}_left{ID}_right{ID}.png
```

File names bind each image to a benchmark question: `modality` (`visual_language`),
`category` (`age`, `gender`, `race`, `religion`), the question/context ids `q…`/`c…` of the
core metadata, and the ids of the source photos placed on the left and right. The harness
resolves them with the glob `./data/images/real_world_image/{modality}_{category}_q*_c*_*.png`
(see `_resolve_image` in `multibbq/inference.py`). Provenance of the source photos is
listed in `data/construction/real_world_images.csv` in the code repository, and the assembly notebook is
`notebooks/gen_realworld.ipynb`.

## Use it with the toolkit

```bash
pip install "multibbq[hf]"
multibbq download --realworld           # places the tree at ./data/images/real_world_image/
multibbq run "OpenGVLab/InternVL3_5-8B" --experiment realworld
```

The corresponding model outputs and computed metrics are in
[MLL-Lab/MultiBBQ-results](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results)
(directories named `gpt_image_gen_realworld`).

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
