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

This section is the single reference for getting the data; everything else links here.

**Already in the repo** (nothing to download): the metadata driving every experiment
(`data/…/mmbbq_*.json`), the construction templates (`data/templates/`), a small image
preview (`data/images_sample/`), and the blank-canvas control image
(`images/pure_white_1024_1024.png`).

**The image sets** live on the HuggingFace Hub. From the **repo root** (inference resolves
`./images/` relative to the current directory):

```bash
pip install -e ".[hf]"
multibbq download                     # main image set                  (~2.7 GB)
multibbq download --realworld         # + real photos                   (~130 MB)
multibbq download --perturbations     # + 11 perturbed sets             (~16 GB)
multibbq download --all               # everything                      (~19 GB)
```

| Group | Source (HF dataset) | Lands at | Needed by |
|---|---|---|---|
| main (default) | [`MLL-Lab/MultiBBQ`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) - extracted from the parquet, byte-identical | `images/gpt_image_gen/`, `images/imagen4ultra_image_gen/` | every image experiment |
| `--realworld` | [`MLL-Lab/MultiBBQ-realworld`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld) | `images/real_world_image/` | `realworld` |
| `--perturbations` | [`MLL-Lab/MultiBBQ-perturbations`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) | `images/gpt_image_gen_<type>/` | `aug_img`, `img_label` |
| blank canvas | ships with the repo | `images/pure_white_1024_1024.png` | `unmasked_wo_img` |

Practical notes:

- **Idempotent / resumable**: re-running skips files already on disk and resumes
  interrupted downloads. `--root <dir>` writes the `images/` tree elsewhere.
- **Cache**: the main-set parquet shards land in `~/.cache/huggingface/` (~2.7 GB,
  relocate with `HF_HOME`); they can be deleted after extraction - `./images/` stands
  on its own.
- **Text-only LLM** evaluation (`--experiment llm`) needs **no images**.

Alternative access, without the toolkit:

```bash
# analysis in Python (embedded images; no ./images/ tree involved)
python -c 'from datasets import load_dataset; ds = load_dataset("MLL-Lab/MultiBBQ", "gpt_image_gen_visual_language", split="test")'

# the two raw-tree repos, via the HF CLI (same landing paths as above)
huggingface-cli download MLL-Lab/MultiBBQ-realworld     --repo-type dataset --local-dir ./images
huggingface-cli download MLL-Lab/MultiBBQ-perturbations --repo-type dataset --local-dir ./images
```

The main set has no raw tree on the Hub (it is embedded in the parquet); a ~15-line
self-serve extraction snippet, the Hub repo layout, and the maintainer build/upload flow
are in [hf.md](../huggingface/hf.md).

To **regenerate** images instead of downloading, see
[dataset-construction.md](../benchmark/dataset-construction.md) (non-deterministic).

## Verify

```bash
multibbq --version
multibbq run "Qwen/Qwen2.5-7B-Instruct" --experiment llm --ambiguous true --negative true
multibbq score --input results/gpt_image_gen_llm/Qwen/Qwen2.5-7B-Instruct/*.json
```

The text-only `llm` smoke test needs no images or API keys, so it is the quickest way to
confirm the install.
