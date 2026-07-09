# Installation

MultiBBQ runs on **Python ≥ 3.9**. Model backends (torch, transformers, model-specific
libraries) are heavy and hardware-dependent, so they install via conda; the package itself
and its metric pipeline are a light `pip install`.

## Full install (inference + metrics)

```bash
# 1. model backends -> conda env `multibbq`
conda env create -f environment.yml
conda activate multibbq

# 2. the multibbq package + CLI
pip install -e .
```

This gives you the `multibbq` command (`run` / `score` / `combine` / `aggregate` /
`pipeline`).

## Metrics-only install (no GPU)

The metric subcommands need only pandas. On a machine without a GPU stack:

```bash
pip install -e .        # pulls in pandas + tqdm only
multibbq score --input results/...
```

`multibbq run` imports the model stack lazily, so it is fine for the model backends to be
absent when you only score.

## Models

Open-source backbones are **HuggingFace ids** and auto-download on first run to
`~/.cache/huggingface` (set `HF_HOME` to relocate; ~3 GB for a 1–2B model up to ~150 GB for
72B). `scripts/download_models.sh` pre-fetches the open-source checkpoints. On offline /
air-gapped clusters, pre-fetch on a login node and set `HF_HUB_OFFLINE=1`.

### API credentials

Set only the keys for the model families you actually run:

| Provider | Models | Environment variable(s) |
|---|---|---|
| OpenAI | GPT-4o, GPT-5 / mini / nano | `OPENAI_API_KEY` |
| Google (Vertex AI) | Gemini 2.5 flash / flash-lite | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `global`); auth via `gcloud auth application-default login` |
| HuggingFace | gated open weights (if any) | `HF_TOKEN` (or `huggingface-cli login`) |

```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_CLOUD_PROJECT=my-vertex-project
```

The full model list and download links are in [models.md](models.md).

## Dataset & images

- **Metadata** (`data/`) and a small **image preview** (`data/images_sample/`) ship in the repo.
- The **full image set** is released on the HuggingFace Hub (`MLL-Lab/MultiBBQ`; see
  [hf.md](hf.md)). The quickest way to get it:

  ```bash
  pip install "multibbq[hf]"
  multibbq download        # writes ./images/ and the auxiliary archives
  ```

  Inference reads images from `./images/` (each record's `image_path` points there). If you
  fetch the archives manually instead, extract them at the repo root to this layout:

  ```
  images/
  ├── gpt_image_gen/{visual,textual}/…
  ├── imagen4ultra_image_gen/{visual,textual}/…
  ├── gpt_image_gen_{noise,brightness_up,…}/…      # perturbed sets (aug_img)
  ├── real_world_image/…                   # real faces (realworld)
  └── pure_white_1024_1024.png             # blank canvas (unmasked_wo_img)
  ```

- To **regenerate** images instead of downloading, see
  [dataset-construction.md](dataset-construction.md) (non-deterministic).
- **Text-only LLM** evaluation (`--experiment llm`) needs **no images**.

## Verify

```bash
multibbq --version
multibbq run "Qwen/Qwen2.5-7B-Instruct" --experiment llm --ambiguous true --negative true
multibbq score --input results/gpt_image_gen_llm/Qwen/Qwen2.5-7B-Instruct/*.json
```

The text-only `llm` smoke test needs no images or API keys, so it is the quickest way to
confirm the install.
