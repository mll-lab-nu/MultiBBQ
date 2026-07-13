# HuggingFace organization

MultiBBQ is released across five homes, kept separate by role, and grouped in one
[HuggingFace collection](https://huggingface.co/collections/MLL-Lab/multibbq-6a4ffcb8ed1d5015cd3f59a6).

| Artifact | Where | Why |
|---|---|---|
| Code (this repo) | GitHub [`mll-lab-nu/MultiBBQ`](https://github.com/mll-lab-nu/MultiBBQ) | the package, CLI, metadata (`data/`), and docs |
| Dataset (metadata + images) | HF Dataset [`MLL-Lab/MultiBBQ`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) | pure parquet; `load_dataset` + viewer, and the source `multibbq download` extracts from |
| Real-world images | HF Dataset [`MLL-Lab/MultiBBQ-realworld`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-realworld) | the real-photo set for the `realworld` experiment (~130 MB) |
| Perturbation images | HF Dataset [`MLL-Lab/MultiBBQ-perturbations`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) | the 11 robustness image sets (~16 GB) |
| Experimental outputs (`results/`, `analysis/`) | HF Dataset [`MLL-Lab/MultiBBQ-results`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results) | reproduction artifacts, browsable on the Hub |

Note the GitHub org is `mll-lab-nu`, while the HuggingFace org is `MLL-Lab`.

## The dataset repo

Pure parquet: four `load_dataset`-able configs, one per generator and modality. Each row
carries the text fields plus its **embedded** image, which powers one-line loading and the
Hub dataset viewer (useful for a fairness benchmark, since the bias images are browsable
for transparency). There are no loose image files here - the raw trees live in the two
companion repos above, and the main tree is re-created locally by `multibbq download`.

```
MLL-Lab/MultiBBQ  (HF dataset)
├── README.md                              # the dataset card (docs/huggingface/hf_dataset_card.md)
├── gpt_image_gen_visual_language/         # config: 410 rows, text fields + category + image
├── gpt_image_gen_visual_only/             # config: 408 rows (2 generator refusals;
│                                          #  the local data/ metadata keeps all 410)
├── imagen4ultra_image_gen_visual_language/
└── imagen4ultra_image_gen_visual_only/
```

Loading for analysis:

```python
from datasets import load_dataset
ds = load_dataset("MLL-Lab/MultiBBQ", "gpt_image_gen_visual_language", split="test")
ds[0]["image"]        # PIL image, ds[0]["image_path"] is the harness path
```

## Using the dataset with the code (`multibbq download`)

The inference harness does not read parquet: it reads **image files from disk** at each
record's `image_path` (e.g. `./data/images/gpt_image_gen/visual_language/...png`), relative to the
directory you run from. `multibbq download` creates that tree: the main set is
**extracted from the parquet shards** (raw bytes, byte-identical to the released images),
and the realworld / perturbation sets are plain file downloads from their repos.

**The complete download guide** - commands, sizes, landing paths, per-experiment needs,
cache notes - lives in one place:
[installation.md Dataset & images section](../getting-started/installation.md#dataset--images).

One Hub-side detail belongs here: the main set has no raw tree on the Hub (the parquet is
the single source), so getting it is always **download the parquet, then extract**.
`multibbq download` does both in one command. Without installing the toolkit, use the
standalone script [`scripts/extract_images.py`](../../scripts/extract_images.py)
(needs only `pip install huggingface_hub pyarrow`; it mirrors `_extract_primary` in
[`multibbq/hf.py`](../../multibbq/hf.py)):

```bash
pip install huggingface_hub pyarrow
python scripts/extract_images.py            # -> ./data/images/  (~2.7 GB)
```

It reuses the standard HF cache, so parquet you already pulled (e.g. with
`huggingface-cli download MLL-Lab/MultiBBQ --repo-type dataset`) is not downloaded twice -
the script goes straight to extraction. Output files are byte-identical to the released
images, and re-running resumes an interrupted extraction.

## Building and pushing the dataset

Maintainers rebuild everything from the repository's canonical `data/` plus a local image
snapshot with [`../../scripts/build_hf_dataset.py`](../../scripts/build_hf_dataset.py):

```bash
pip install "multibbq[hf]"
huggingface-cli login                  # or set HF_TOKEN
python scripts/build_hf_dataset.py --source <snapshot>          # build + verify only
python scripts/build_hf_dataset.py --source <snapshot> --push   # + upload
```

The script embeds each image into sharded parquet, assembles the staging tree (four configs
+ the card), and uploads it to `MLL-Lab/MultiBBQ` with `upload_folder`; it then uploads the
raw `real_world_image/` tree to `MLL-Lab/MultiBBQ-realworld` and the perturbation sets to
`MLL-Lab/MultiBBQ-perturbations` (`--skip-core` / `--skip-realworld` /
`--skip-perturbations` select the parts). It deliberately avoids `Dataset.push_to_hub`
(whose card-merge step can raise a spurious `UnicodeDecodeError`). The card's YAML front
matter declares the license, tags, and config layout for the viewer.

## The outputs repo

The raw model outputs (`results/`) and computed metrics (`analysis/`) are reproduction
artifacts, not a dataset to train on, so they live in a separate HF dataset repo. Upload with
`upload_folder` (or the CLI):

```bash
huggingface-cli upload MLL-Lab/MultiBBQ-results results/  results/  --repo-type dataset
huggingface-cli upload MLL-Lab/MultiBBQ-results analysis/ analysis/ --repo-type dataset
```

Everything in `analysis/` is regenerable from `results/` with `multibbq pipeline`, so the
outputs repo is a convenience and an audit trail. See [RESULTS.md](../benchmark/RESULTS.md) and the repo
card [`hf_results_card.md`](hf_results_card.md).
