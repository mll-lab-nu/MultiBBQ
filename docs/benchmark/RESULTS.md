# Experimental Outputs & Images

To keep the repository lightweight, large artifacts are **not** stored in git. They live on
the HuggingFace Hub and are reproducible from the code here. The full layout is in
[hf.md](../huggingface/hf.md).

| Artifact | Size | Where | How to regenerate |
|----------|------|-------|-------------------|
| Main image set (`data/images/{gpt,imagen4ultra}_image_gen/`) | ~2.7 GB | HF dataset `MLL-Lab/MultiBBQ` (embedded in parquet) | `notebooks/gen_images_*.ipynb` |
| Real-world images (`data/images/real_world_image/`) | ~130 MB | HF dataset `MLL-Lab/MultiBBQ-realworld` | `notebooks/gen_realworld.ipynb` |
| Perturbed image sets (`data/images/gpt_image_gen_<type>/`) | ~16 GB | HF dataset `MLL-Lab/MultiBBQ-perturbations` | perturbation transforms of the main set |
| Raw inference outputs (`results/`) | ~330 MB | HF dataset `MLL-Lab/MultiBBQ-results` | `multibbq run ...` (see main README) |
| Computed metrics (`analysis/`) | ~110 MB | HF dataset `MLL-Lab/MultiBBQ-results` | `multibbq pipeline --input results/ --output analysis/` |

## Getting the images

```bash
pip install -e ".[hf]"
multibbq download          # main set -> ./data/images/; --realworld / --perturbations / --all
```

The complete download guide (all methods, sizes, landing paths) is
[installation.md Dataset & images section](../getting-started/installation.md#dataset--images).

## Reproducing the analysis from raw outputs

Fetch the released outputs (or any slice of them), then run the pipeline:

```bash
hf download MLL-Lab/MultiBBQ-results --repo-type dataset --include "results/gpt_image_gen_main/**" --local-dir .
pip install -e .
multibbq pipeline --input results/gpt_image_gen_main --output analysis/gpt_image_gen_main
```

The released `results/` cover the experiments reported in the paper; the toolkit's
extension settings (`img_label`, `context_unmasked`, `llm`) and the perturbation
baselines (`brightness`, `contrast`) have no released outputs.

Only the dataset metadata (`data/metadata/`), the construction materials
(`data/construction/`), and the small preview (`data/images_sample/`) are tracked in git.

## Experiment ↔ paper map

| `--experiment` | Paper section | Historical script (pre-release codebase) |
|----------------|---------------|------------------------------------------|
| `main` | Main results | `eval_models.py` |
| `aug_img` | Impact of image quality | `eval_models_aug_img.py` |
| `img_label` | (control, not reported in the paper) | `eval_models_img_label.py` |
| `quant` | Impact of quantization | `eval_models_quant.py` |
| `temp` | Impact of decoding temperature | `eval_models_temp.py` |
| `reasoning` | Bias mitigation | `eval_models_reasoning.py` |
| `realworld` | Generalization to real images | `eval_models_realworld.py` |
| `context_unmasked` | (control, not reported in the paper) | `eval_models_context_unmasked.py` |
| `unmasked_w_img` / `unmasked_wo_img` | Backbone-LLM comparison | `eval_models_unmasked_{w,wo}_img.py` |
