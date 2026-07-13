---
license: cc-by-4.0
pretty_name: MultiBBQ
task_categories:
- visual-question-answering
- question-answering
language:
- en
tags:
- fairness
- social-bias
- multimodal
- vision-language
- bbq
size_categories:
- 1K<n<10K
configs:
- config_name: gpt_image_gen_visual_language
  data_files:
  - split: test
    path: gpt_image_gen_visual_language/test-*
- config_name: gpt_image_gen_visual_only
  data_files:
  - split: test
    path: gpt_image_gen_visual_only/test-*
- config_name: imagen4ultra_image_gen_visual_language
  data_files:
  - split: test
    path: imagen4ultra_image_gen_visual_language/test-*
- config_name: imagen4ultra_image_gen_visual_only
  data_files:
  - split: test
    path: imagen4ultra_image_gen_visual_only/test-*
---

<br>

<p align="center">
  <img src="https://huggingface.co/datasets/MLL-Lab/MultiBBQ/resolve/main/logo_horizontal.png" alt="MultiBBQ logo" width="620"/>
</p>

<br>

<h1 align="center">MultiBBQ: A Fairness Benchmark for Multimodal LLMs</h1>

<p align="center">
  <em>Controllable diagnosis of social bias in multimodal LLMs with synthetic images.</em>
</p>

MultiBBQ is a fairness evaluation benchmark for multimodal large language models (MLLMs).
It extends the language-only [BBQ](https://github.com/nyu-mll/BBQ) benchmark into the visual
domain: each attested social bias is paired with an AI-generated photorealistic image of two
people who differ **only in the target demographic**, so a model's fairness can be probed
when the demographic signal is carried by the **image** rather than by text. The design gives
precise control over the demographic configuration in every example, which is difficult to
achieve with in-the-wild photos.

<p align="center">
  <img src="https://huggingface.co/datasets/MLL-Lab/MultiBBQ/resolve/main/multibbq_example.png" alt="A MultiBBQ example: one image pair evaluated under visual-only ambiguous, visual-language ambiguous, and visual-language disambiguated contexts, with positional answer options" width="760"/>
</p>

Each example is evaluated under **three scenarios**, and fair behavior is well defined in
each:

| Scenario | The model sees | Fair behavior |
|---|---|---|
| **Visual-Only, Ambiguous** | image only | answer **Unknown**: the image alone supports neither person |
| **Visual-Language, Ambiguous** | image + under-informative context | answer **Unknown** |
| **Visual-Language, Disambiguated** | image + context that determines the answer | pick the evidence-backed person, whether or not that aligns with the stereotype |

- **Paper:** *Fairness Failure Modes of Multimodal LLMs*. This work is honored to receive the 🏆 **[Best Paper Award](https://drive.google.com/file/d/1OZcaRvlcB6uqkRgm5ve-ds0xS4TuW_6Z/view?usp=sharing)** in the *ACL 2026 Workshop on Trustworthy Natural Language Processing*.
- **Code (evaluation toolkit):** https://github.com/mll-lab-nu/MultiBBQ
- **Project page:** https://multibbq.github.io
- **Companion repos:** [MultiBBQ-realworld](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld) (real-photo transferability set) · [MultiBBQ-perturbations](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) (robustness image sets) · [MultiBBQ-results](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results) (model outputs + metrics)
- **License:** CC-BY-4.0 (dataset). Code is MIT.

## What is in this repo

This repo is the core benchmark, and it is **pure parquet**: four `load_dataset`-able
configs with every image embedded. There are no loose image files here.

```
MLL-Lab/MultiBBQ
├── gpt_image_gen_visual_language/          # config: GPT-Image-1, image + text context
├── gpt_image_gen_visual_only/              # config: GPT-Image-1, image only
├── imagen4ultra_image_gen_visual_language/ # config: Imagen 4 Ultra, image + text context
└── imagen4ultra_image_gen_visual_only/     # config: Imagen 4 Ultra, image only
```

The auxiliary image sets live in companion repos:
[MultiBBQ-realworld](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld) (real
photos, transferability experiment) and
[MultiBBQ-perturbations](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations)
(11 perturbed variants, robustness experiments). The blank-canvas control image and the
text metadata ship with the [code repository](https://github.com/mll-lab-nu/MultiBBQ).

## Composition

410 examples and 2,460 question-answer pairs, across four demographic categories chosen so
that the target attribute is visually identifiable in a portrait.

| Category | Examples | QA pairs | Subgroups |
|---|---|---|---|
| Race | 127 | 762 | Black, White, East Asian, Native American, Hispanic, Arab |
| Gender | 50 | 300 | Male, Female |
| Religion | 134 | 804 | Christian, Hindu, Muslim, Buddhist, Jewish, Orthodox, Atheist |
| Age | 99 | 594 | Young, Middle-aged, Old Adult |

Each example is instantiated as multiple QA pairs by crossing two context conditions
(**ambiguous**, where the correct answer is *Unknown*, and **disambiguated**, where the
context determines a specific answer) with two question framings (**negative** and
**non-negative**), following the BBQ protocol.

## Two image generators, two modalities

The same 410 examples are rendered by two independent image generators, and each is provided
in two modalities. This yields the four configs above.

- **Image generators.** `GPT-Image-1` and `Imagen 4 Ultra`. Running both lets you check that
  a model's measured bias ranking reflects the model, not one generator's visual style. The
  paper reports very high cross-generator agreement.
- **Visual-language (VL).** The image is accompanied by the textual context and question.
- **Visual-only (VO).** The image carries the demographic evidence and the text is stripped
  of it; answer options use positional references (for example, "the person on the left").

| Config | Generator | Modality | Rows |
|---|---|---|---|
| `gpt_image_gen_visual_language` | GPT-Image-1 | visual-language | 410 |
| `gpt_image_gen_visual_only` | GPT-Image-1 | visual-only | 408 |
| `imagen4ultra_image_gen_visual_language` | Imagen 4 Ultra | visual-language | 410 |
| `imagen4ultra_image_gen_visual_only` | Imagen 4 Ultra | visual-only | 408 |

The two visual-only configs have 408 rows rather than 410: for two examples the image
generator declined to produce the visual-only image (a content-policy refusal), so it is not
part of the released set. The visual-language configs are complete at 410.

## Schema

Every row carries the full BBQ-style text metadata plus the embedded image. Selected fields:

| Field | Type | Description |
|---|---|---|
| `category` | string | `race` / `gender` / `religion` / `age` |
| `q_id`, `c_id` | int | question id and context id within the category |
| `ambig_context`, `disambig_context` | string | ambiguous and disambiguated context sentences |
| `ambig_context_masked`, `disambig_context_masked` | string | same contexts with demographic terms replaced by positional references |
| `neg_q`, `nonneg_q` | string | negative and non-negative question |
| `options`, `options_masked` | list | answer options (plain / positional) |
| `neg_label_*`, `nonneg_label_*`, `unk_label_idx` | int / string | gold labels for each framing and the Unknown option |
| `stereotype_group_*`, `nonstereotype_group_*` | int / string | the stereotyped and non-stereotyped subgroups in this pair |
| `person_on_the_left`, `person_on_the_right` | string | which subgroup is where in the image |
| `visual_only_ambig_prompt_w_position`, `..._wo_position` | string | prompts used in the visual-only condition |
| `visual_textual_prompt` | string | prompt used in the visual-language condition |
| `image_path` | string | original harness-relative path, for example `./images/gpt_image_gen/textual/...png` |
| `image` | image | the embedded PNG (1024x1024) |

## Load it

```python
from datasets import load_dataset

ds = load_dataset("MLL-Lab/MultiBBQ", "gpt_image_gen_visual_language", split="test")
row = ds[0]
print(row["category"], row["options"])
row["image"]            # PIL.Image
```

Swap the config name for any of the four subsets.

## Evaluate a model with the toolkit

The MultiBBQ harness reads images from local paths (`./images/...`). `multibbq download`
re-creates that tree from this repo: it pulls the parquet shards and writes each row's
embedded PNG back to its `image_path` — byte-identical to the released images. The harness
evaluates both vision-language models and, on the unmasked text, text-only LLMs.

```bash
git clone https://github.com/mll-lab-nu/MultiBBQ && cd MultiBBQ
pip install -e ".[hf]"
multibbq download                                  # main image set -> ./images/ (~2.7 GB)
multibbq run "OpenGVLab/InternVL3_5-8B" --experiment main
multibbq pipeline --input results/... --output analysis/...   # Fairness / Bias / Unknown-rate
```

`multibbq download --realworld` / `--perturbations` additionally fetch the companion image
sets when you run those experiments.

## Metrics (summary)

Model responses are scored with three modality-agnostic measures:

- **Fairness Score (FS, higher is better)** rewards choosing *Unknown* when the context is
  ambiguous and the correct answer when it is disambiguated.
- **Bias Score (BS, lower is better)** measures how far answers skew toward the stereotyped
  subgroup.
- **Unknown-rate** tracks abstention behavior.

`FS_Total` and `BS_Total` combine three scenarios (visual-only ambiguous, visual-language
ambiguous, visual-language disambiguated) via a harmonic mean. See the code docs and the
paper for exact definitions.

## Key designs

- **Shortcut Mitigation.** MLLMs tend to over-rely on text and neglect the image. If the
  question or options contain demographic terms ("the man", "the woman"), a model can pick
  the correct answer from language alone without reasoning over the image. MultiBBQ replaces
  demographic terms with **positional references** ("the person on the left / right") that
  only the image can resolve, enforcing cross-modal reasoning. Disambiguated contexts
  deliberately keep their demographic terms: there, mapping the description to a position
  still requires the image. Option order and the stereotype / non-stereotype assignment are
  randomized to remove position and ordering shortcuts.
- **Controllable image synthesis.** Synthetic images make each pair a controlled experiment
  (only the target demographic differs), avoid training-data contamination, and involve no
  real individuals. Every image passed a **four-rater, all-pass quality filter** for
  **Identifiability**, **Faithfulness**, and **Controllability**. Model rankings agree
  across the two generators (Pearson r = 0.9963 on FS_Total) and transfer to real face
  images (r = 0.9787).
- **Construction.** Templates are adapted from BBQ (names and visually ambiguous subgroups
  pruned) and demographic cues are moved into the image. Full details are in the paper and
  in the code repo's `docs/dataset-construction.md`.

## Intended use and limitations

- Intended for **evaluation** of model fairness, not for training.
- Images are synthetic; they support variable control and avoid using real individuals, but
  they are portraits and do not cover in-the-wild scenes or multi-person crowds.
- Demographic subgroups are a curated, non-exhaustive taxonomy inherited from BBQ.
- Two visual-only examples are absent due to generator content-policy refusals (see above).

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

MultiBBQ is built on the BBQ benchmark ([github.com/nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)); please also cite it:

```bibtex
@inproceedings{parrish2022bbq,
  title     = {{BBQ}: A Hand-Built Bias Benchmark for Question Answering},
  author    = {Parrish, Alicia and Chen, Angelica and Nangia, Nikita and
               Padmakumar, Vishakh and Phang, Jason and Thompson, Jana and
               Htut, Phu Mon and Bowman, Samuel R.},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
  year      = {2022},
  url       = {https://arxiv.org/abs/2110.08193},
}
```
