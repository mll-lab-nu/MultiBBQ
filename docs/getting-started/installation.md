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

The full model list and download links are in [models.md](../benchmark/models.md).

## Dataset & images

- **Metadata** (`data/`), a small **image preview** (`data/images_sample/`), and the
  **blank-canvas control image** (`images/pure_white_1024_1024.png`) ship in the repo.
- The **full image set** is released on the HuggingFace Hub (see
  [hf.md](../huggingface/hf.md) for the repo layout and all download options). Get it with:

  ```bash
  pip install -e ".[hf]"
  multibbq download                     # main image set -> ./images/   (~2.7 GB)
  multibbq download --realworld         # + real photos, `realworld` experiment (~130 MB)
  multibbq download --perturbations     # + perturbed sets, `aug_img`/`img_label` (~16 GB)
  ```

  Run it from the repo root: inference reads images from `./images/` relative to the
  current directory (each record's `image_path` points there). The command is idempotent —
  re-run it to resume or verify. The default download is enough for every experiment
  except `realworld`, `aug_img`, and `img_label`. After the download you have:

  ```
  images/
  ├── gpt_image_gen/{visual,textual}/…             # default
  ├── imagen4ultra_image_gen/{visual,textual}/…    # default
  ├── real_world_image/…                           # --realworld
  ├── gpt_image_gen_{noise,brightness_up,…}/…      # --perturbations
  └── pure_white_1024_1024.png                     # already in the repo (unmasked_wo_img)
  ```

- To **regenerate** images instead of downloading, see
  [dataset-construction.md](../benchmark/dataset-construction.md) (non-deterministic).
- **Text-only LLM** evaluation (`--experiment llm`) needs **no images**.

## Verify

```bash
multibbq --version
multibbq run "Qwen/Qwen2.5-7B-Instruct" --experiment llm --ambiguous true --negative true
multibbq score --input results/gpt_image_gen_llm/Qwen/Qwen2.5-7B-Instruct/*.json
```

The text-only `llm` smoke test needs no images or API keys, so it is the quickest way to
confirm the install.
