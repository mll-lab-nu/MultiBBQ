# Experiments

Every evaluation setting is one `--experiment` value, driven by the `EXPERIMENTS` table in
[`../multibbq/experiments.py`](../multibbq/experiments.py). An experiment fixes a
combination of orthogonal axes (image source, text-field masking, option formatting, model
mode) and a results subdirectory token.

| `--experiment` | Study | Extra flags | Paper section | Legacy script |
|---|---|---|---|---|
| `main` | Baseline (gpt_image_gen / imagen4ultra_image_gen) | (none) | Main results | `eval_models.py` |
| `aug_img` | Image perturbation | `--img_aug_type` | Impact of image quality | `eval_models_aug_img.py` |
| `img_label` | Generic-label options (`person A/B/C`) | `--img_aug_type label` | Option-format control | `eval_models_img_label.py` |
| `quant` | Model quantization | (none) | Impact of quantization | `eval_models_quant.py` |
| `temp` | Decoding temperature | `--temperature` | Impact of decoding temperature | `eval_models_temp.py` |
| `reasoning` | Reasoning / fairness instruction | `--reasoning_mode` | Bias mitigation | `eval_models_reasoning.py` |
| `realworld` | Real-world images (VL only) | (none) | Generalization to real images | `eval_models_realworld.py` |
| `context_unmasked` | Demographic names injected into context | (none) | Language-leakage control | `eval_models_context_unmasked.py` |
| `unmasked_w_img` | Backbone eval, unmasked text + image | (none) | Backbone-LLM comparison | `eval_models_unmasked_w_img.py` |
| `unmasked_wo_img` | Backbone eval, unmasked text, blank image | (none) | Backbone-LLM comparison | `eval_models_unmasked_wo_img.py` |
| `llm` | **Text-only LLM evaluation** (no image) | (none) | (extension) | (none) |

## Axes (how the settings differ)

- **fields**: `masked` uses demographic-neutral columns (`*_masked`), `unmasked` uses the
  raw columns. Masking is MultiBBQ's language-leakage control.
- **image**: `dataset` (the generated image), `aug` (a perturbed variant under
  `gpt_image_gen_<type>/`), `blank` (a white canvas, for `unmasked_wo_img`), `realworld`, or
  `none` (text-only `llm`).
- **options**: `plain` shows the (masked) choices; `label` replaces the non-answer choices
  with `person A/B/C`.
- **mode**: the model wrapper mode; `reasoning` is resolved from `--reasoning_mode`.
- **text_only / strip_image_ref**: set only for `llm`, to run a text LLM with no image, on the
  unmasked text, with `" in the image"` stripped from the question. See
  [llm-evaluation.md](llm-evaluation.md).

The four `reasoning_mode` values combine a **reasoning** vs **non-reasoning** system
instruction with an optional **fairness** instruction, isolating the effect of each
mitigation. `temp` sweeps decoding temperature, and `aug_img` sweeps the perturbation type.

> The legacy scripts are the pre-release per-experiment entry points, kept in the sibling
> `multibbq_legacy/` folder for cross-checking; the unified loop reproduces each one
> exactly. Adding a new experiment: [extending.md](extending.md).
