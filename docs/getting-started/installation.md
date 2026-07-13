# Installation

MultiBBQ runs on **Python ≥ 3.9**. Model backends (torch, transformers, model-specific
libraries) are heavy and hardware-dependent, so they install via conda; the package itself
and its metric pipeline are a light `pip install`.

## Full install (inference + metrics)

The inference environment targets **Linux with an NVIDIA GPU** (the pinned torch wheels
are CUDA 12.4 builds). API-only inference (GPT / Gemini) and the metric pipeline do not
need a GPU.

```bash
# 1. model backends -> conda env `multibbq`
conda env create -f environment.yml
conda activate multibbq
pip install flash-attn==2.7.4.post1 --no-build-isolation   # builds against the installed torch

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
| HuggingFace | gated open weights (if any) | `HF_TOKEN` (or `hf auth login`) |

```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_CLOUD_PROJECT=my-vertex-project
```

The full model list and download links are in [models.md](../benchmark/models.md).

## Dataset & images

This section is the single reference for getting the data; everything else links here.

Everything an experiment reads lives under `data/`. Part of it ships with the git clone,
part is downloaded:

| What | Path | How you get it |
|---|---|---|
| Metadata (the questions, contexts, labels) | `data/metadata/…/multibbq_*.json` | ships with the clone |
| Construction templates + image preview | `data/construction/templates/`, `data/images_sample/` | ships with the clone |
| Blank-canvas control image | `data/images/pure_white_1024_1024.png` | ships with the clone |
| **Main / real-world / perturbation images** | `data/images/…` | **download, one of the methods below** |

### Method A - `multibbq download` (recommended)

**Step 1.** Install the download extras (once):

```bash
pip install -e ".[hf]"
```

**Step 2.** From the **repo root** (inference resolves `./data/images/` relative to the
current directory), fetch the group(s) you need:

```bash
multibbq download                     # main image set                  (~2.7 GB)
multibbq download --realworld         # + real photos                   (~130 MB)
multibbq download --perturbations     # + 11 perturbed sets             (~16 GB)
multibbq download --all               # everything                      (~19 GB)
```

| Flag | Source (HF dataset) | Lands at | Needed by |
|---|---|---|---|
| (default) | [`MLL-Lab/MultiBBQ`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) - extracted from the parquet, byte-identical | `data/images/gpt_image_gen/`, `data/images/imagen4ultra_image_gen/` | every image experiment |
| `--realworld` | [`MLL-Lab/MultiBBQ-realworld`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld) | `data/images/real_world_image/` | `realworld` |
| `--perturbations` | [`MLL-Lab/MultiBBQ-perturbations`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) | `data/images/gpt_image_gen_<type>/` | `aug_img`, `img_label` |
| (nothing) | blank canvas, already in the repo | `data/images/pure_white_1024_1024.png` | `unmasked_wo_img` |

**Step 3.** Check what you got. The default run ends with:

```
[primary] 1636 images written, 0 already present
[download] done -> ./data/images
```

and afterwards:

```bash
find data/images -name "*.png" | wc -l    # 1637 (1636 downloaded + the blank canvas)
```

Good to know:

- **Idempotent / resumable**: re-running skips files already on disk and resumes
  interrupted downloads; if unsure, just run it again.
- **408 vs 410**: 4 metadata records (2 per generator, visual-only) reference images
  the generators refused to produce, so visual-only runs cover 408 of the 410 items;
  the harness prints `Image not found` for those records and skips them.
- **`--root <dir>`** writes the `data/images/` tree under another directory.
- **Cache**: the main-set parquet shards land in `~/.cache/huggingface/` (~2.7 GB,
  relocate with `HF_HOME`); they can be deleted after extraction, since
  `./data/images/` stands on its own.
- **Text-only LLM** evaluation (`--experiment llm`) needs **no images**.

### Method B - HuggingFace CLI (no toolkit needed)

The real-world and perturbation repos are raw file trees, so the plain `hf` CLI (ships
with `pip install huggingface_hub`) drops them straight into place (same landing paths
as Method A):

```bash
hf download MLL-Lab/MultiBBQ-realworld     --repo-type dataset --local-dir ./data/images
hf download MLL-Lab/MultiBBQ-perturbations --repo-type dataset --local-dir ./data/images
```

The **main set** has no raw tree on the Hub: it is embedded in the parquet. The plain CLI
can download the parquet (`hf download MLL-Lab/MultiBBQ --repo-type dataset`),
but that is not yet the image tree the harness needs — one extraction step turns it into
one. Use Method A (which downloads *and* extracts in one command), or the standalone
script [`scripts/extract_images.py`](../../scripts/extract_images.py) (needs only
`huggingface_hub` + `pyarrow`, not the toolkit):

```bash
python scripts/extract_images.py            # download (cache-aware) + extract -> ./data/images/
```

It reuses the HF cache, so parquet you already downloaded is not fetched again.

### Method C - `load_dataset` (analysis only)

```python
from datasets import load_dataset
ds = load_dataset("MLL-Lab/MultiBBQ", "gpt_image_gen_visual_language", split="test")
ds[0]["image"]   # PIL image; metadata fields alongside
```

This serves notebooks and analysis: images arrive embedded, nothing is written to
`data/images/`, and the evaluation harness will **not** find them. To run evaluations,
use Method A or B.

### The result, on disk

```
data/images/
├── gpt_image_gen/{visual_language,visual_only}/…             # default download
├── imagen4ultra_image_gen/{visual_language,visual_only}/…    # default download
├── real_world_image/…                           # --realworld
├── gpt_image_gen_{noise,brightness_up,…}/…      # --perturbations
└── pure_white_1024_1024.png                     # ships with the repo
```

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
