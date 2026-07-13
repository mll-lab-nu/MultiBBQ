# Dataset Construction

MultiBBQ turns the text-only [BBQ](https://github.com/nyu-mll/BBQ) benchmark
(Parrish et al.) into a **multimodal** fairness benchmark: each question is paired with a
generated image, and the language is rewritten so a model cannot answer from linguistic
shortcuts alone. Everything below is reproducible from the provenance notebooks in
[`../../notebooks/`](../../notebooks/) (they document construction; they are **not** the
evaluation path, which is described in [`running.md`](../getting-started/running.md)).

## Overview

We adopt BBQ's four social-bias categories and their subgroups, then adapt them for the
visual setting: we **exclude** the personal-name templates and **consolidate**
visually-redundant subgroups (e.g. *Hispanic* subsumes *Latino / Latina / Latin
American*). The result is **410 examples / 2,460 QA pairs** (one negative and one
non-negative question per ambiguous/disambiguated context).

| Category | Examples | QA pairs |
|---|---:|---:|
| Race | 127 | 762 |
| Gender | 50 | 300 |
| Religion | 134 | 804 |
| Age | 99 | 594 |
| **Total** | **410** | **2,460** |

The construction pipeline runs in this order:

1. **Text**: [`gen_template.ipynb`](../../notebooks/gen_template.ipynb) turns templates + vocabulary → the revised record table.
2. **Images**: [`gen_images_gpt_image_gen.ipynb`](../../notebooks/gen_images_gpt_image_gen.ipynb) and [`gen_images_imagen4ultra_image_gen.ipynb`](../../notebooks/gen_images_imagen4ultra_image_gen.ipynb).
3. **Human QC**: a four-evaluator, three-criteria unanimous filter (offline).
4. **Real-world split**: [`gen_realworld.ipynb`](../../notebooks/gen_realworld.ipynb).

## 1. Text construction

[`gen_template.ipynb`](../../notebooks/gen_template.ipynb) slots the demographic vocabulary
into the per-category templates and cleans up the result.

| | |
|---|---|
| **Inputs** | [`../../data/templates/new_templates_*.csv`](../../data/templates/) (Race / Gender / Religion / Age), [`../../data/templates/vocabulary.csv`](../../data/templates/vocabulary.csv), `utils.py` |
| **Output** | [`../../data/multibbq_template_table.csv`](../../data/) (the table every downstream step reads) |

Key transformations applied while building each record:

- **Visual grounding.** Contexts are re-anchored to the accompanying picture: masked
  answers get " in the image" appended so the question refers to the depicted persons.
- **Neutralization.** Gendered pronouns and indirect demographic indicators are replaced
  with a neutral **"the person"**, and the two entities are placed **on the left / on the
  right** so they differ only along the target demographic axis.
- **Masking (shortcut-reasoning mitigation).** In **ambiguous** contexts and their
  options, demographic terms are replaced with **positional** references ("the person on
  the left / right") so a model cannot infer *Unknown* from language alone. In
  **disambiguated** contexts the demographic terms are **retained** in the added
  disambiguating sentence. Each record therefore carries both a `masked` and an
  `unmasked` copy of its context/options.
- **Order-bias mitigation.** Both the **answer-option order** and the
  **assignment/order of the stereotype vs non-stereotype group** are randomized per
  record.
- **Grammar clean-up.** A revision pass fixes the rough slotted strings: raw positional
  masks become natural phrasing ("a left side man" → "the man on the left"), and
  indefinite articles are corrected to the following word's *sound* ("a hour" → "an
  hour"), removing linguistic shortcut cues.

Each row also stores the negative / non-negative / unknown option indices, the
stereotype / non-stereotype indices, and the three image-generation prompts
(`visual_textual_prompt`, `visual_only_ambig_prompt_w_position`,
`visual_only_ambig_prompt_wo_position`).

## 2. Image generation

Two generators produce a **parallel** image set for cross-generator robustness. Each
generator is run in **two variants**: **visual-language** (people carry visible
demographic cues that match the textual scene) and **visual-only** (positional prompts,
no textual cue). Both notebooks read the same
[`../../data/multibbq_template_table.csv`](../../data/) and write generator-specific outputs.

| Generator | Model | Notebook | Credential | Images → | Provenance tables → |
|---|---|---|---|---|---|
| GPT-Image-1 (primary) | `gpt-image-1` | [`gen_images_gpt_image_gen.ipynb`](../../notebooks/gen_images_gpt_image_gen.ipynb) | `OPENAI_API_KEY` | `data/images/gpt_image_gen/{visual_language,visual_only}/` | [`../../data/gpt_image_gen/`](../../data/) |
| Imagen-4-Ultra (parallel) | `imagen-4.0-ultra-generate-001` (Vertex AI) | [`gen_images_imagen4ultra_image_gen.ipynb`](../../notebooks/gen_images_imagen4ultra_image_gen.ipynb) | Vertex ADC + `GOOGLE_CLOUD_PROJECT` | `data/images/imagen4ultra_image_gen/{visual_language,visual_only}/` | [`../../data/imagen4ultra_image_gen/`](../../data/) |

Directory naming matches the modality: `visual_language/` images pair with
`multibbq_visual_language.json` (scenes matching a textual context) and `visual_only/`
images pair with `multibbq_visual_only.json` (positional composition, no textual cue).
Each image is written with a `.txt` sidecar recording its prompt, contexts and Q&A. The
**non-image fields are identical across generators**; only the `image_path` column
differs. Image **perturbations** (the `aug_img` study in
[`experiments.md`](experiments.md)) are applied to the **GPT-Image-1** images only.

> **Determinism caveat.** GPT-Image-1 and Imagen-4-Ultra sampling is **not seed-stable**,
> so re-running these notebooks yields images that differ from the released set. For
> faithful reproduction, use the **released images** (see [`RESULTS.md`](RESULTS.md)),
> not regenerated ones. This is safe because cross-generator ranking correlation is very
> high: Pearson **r = 0.9963** (FS) / **0.9964** (BS) between GPT-Image-1 and Imagen, so
> model rankings are stable even though individual images differ.

## 3. Human quality control

Every image is screened by **four independent evaluators** against **three criteria**:

| # | Criterion | Passes when… |
|---|---|---|
| 1 | **Identifiability** | the target demographic is recognizable in the image |
| 2 | **Faithfulness** | the image matches the textual context (scene, number of persons) |
| 3 | **Controllability** | the two persons differ **only** in the target demographic |

An image is kept **only if all four evaluators mark all three criteria satisfied**
(unanimous all-pass). Failures are regenerated with refined prompts; templates that
cannot be recovered are dropped from the benchmark.

## 4. Real-world image split

For a generalization test on real photographs,
[`gen_realworld.ipynb`](../../notebooks/gen_realworld.ipynb) pairs each item's left/right
entities with real faces from the **Face Research Lab London Set** (adults 18–60;
ethnicity restricted to **Black / White / East Asian**).

| | |
|---|---|
| **Inputs** | [`../../data/multibbq_template_table.csv`](../../data/), the face catalog ([`../../data/real_world_images.csv`](../../data/)) |
| **Output** | [`../../data/multibbq_template_table_w_face_id.csv`](../../data/) (per-item `left_face_id` / `right_face_id`) |

The pairing routine reads each side's masked description and matches race, gender, and
age (age words such as *young / middle-aged / old* map to concrete face-age ranges),
sampling a different face for each side. **Religion** rows get no face (faces don't
encode religion), and any item with no matchable face pair is excluded from the split.
The real-world split is **visual-language only** (the `realworld` experiment).

## 5. Regeneration checklist

Run **from the `notebooks/` folder** (paths are relative: `../../data/templates/…`, `../../data/…`).
The write cells **overwrite the shipped files under `../../data/`**, so change the output path
if you only want to inspect.

1. `python`/Jupyter env with the notebook deps; set `OPENAI_API_KEY` (GPT-Image-1) and
   Vertex ADC + `GOOGLE_CLOUD_PROJECT` (Imagen-4-Ultra).
2. **Text:** run [`gen_template.ipynb`](../../notebooks/gen_template.ipynb) → writes
   [`../../data/multibbq_template_table.csv`](../../data/).
3. **Images:** run [`gen_images_gpt_image_gen.ipynb`](../../notebooks/gen_images_gpt_image_gen.ipynb)
   and [`gen_images_imagen4ultra_image_gen.ipynb`](../../notebooks/gen_images_imagen4ultra_image_gen.ipynb)
   → write images + `../../data/{gpt_image_gen,imagen4ultra_image_gen}/` tables. **(Non-deterministic; see the
   caveat above and prefer the released images.)**
4. **Real-world:** run [`gen_realworld.ipynb`](../../notebooks/gen_realworld.ipynb) → writes
   [`../../data/multibbq_template_table_w_face_id.csv`](../../data/).
5. The image notebooks write to `../data/images/`, i.e. the repository `data/images/` tree,
   so you can evaluate directly with `multibbq run` (see [`running.md`](../getting-started/running.md)).

See also: [`../../notebooks/README.md`](../../notebooks/README.md),
[`../../data/templates/README.md`](../../data/templates/README.md),
[`../../data/README.md`](../../data/README.md).
