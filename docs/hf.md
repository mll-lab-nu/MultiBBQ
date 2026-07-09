# HuggingFace organization

MultiBBQ is released across four homes, kept separate by role, and grouped in one
[HuggingFace collection](https://huggingface.co/collections/MLL-Lab/multibbq-6a4ffcb8ed1d5015cd3f59a6).

| Artifact | Where | Why |
|---|---|---|
| Code (this repo) | GitHub [`mll-lab-nu/MultiBBQ`](https://github.com/mll-lab-nu/MultiBBQ) | the package, CLI, and docs |
| Dataset (metadata + images) | HF Dataset [`MLL-Lab/MultiBBQ`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ) | what people reuse; `load_dataset` + viewer |
| Perturbation images | HF Dataset [`MLL-Lab/MultiBBQ-perturbations`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-perturbations) | the 11 robustness image sets (~16 GB) |
| Experimental outputs (`results/`, `analysis/`) | HF Dataset [`MLL-Lab/MultiBBQ-results`](https://huggingface.co/datasets/MLL-Lab/MultiBBQ-results) | reproduction artifacts, browsable on the Hub |

Note the GitHub org is `mll-lab-nu`, while the HuggingFace org is `MLL-Lab`.

## The dataset repo

A `load_dataset`-able HuggingFace dataset with four configs, one per generator and
modality. Each row carries the text fields plus its embedded image, which powers one-line
loading and the dataset viewer (useful for a fairness benchmark, since the bias images are
browsable for transparency).

```
MLL-Lab/MultiBBQ  (HF dataset)
├── README.md                              # the dataset card (docs/hf_dataset_card.md)
├── gpt_image_gen_visual_language/         # config: 410 rows, text fields + category + image
├── gpt_image_gen_visual_only/             # config: 408 rows (2 generator refusals;
│                                          #  the local data/ metadata keeps all 410)
├── imagen4ultra_image_gen_visual_language/
├── imagen4ultra_image_gen_visual_only/
└── image_archives/                        # auxiliary images with no new metadata
    ├── real_world.tar.gz                  # real_world_image/ for the realworld experiment
    └── pure_white_1024_1024.png           # blank canvas for unmasked_wo_img
```

The image perturbation sets (used by the `aug_img` / `img_label` experiments) are large and
reuse the core metadata, so they live in their own repo, `MLL-Lab/MultiBBQ-perturbations`,
as raw image trees (`gpt_image_gen_<type>/{textual,visual}/`).

Loading:

```python
from datasets import load_dataset
ds = load_dataset("MLL-Lab/MultiBBQ", "gpt_image_gen_visual_language", split="test")
```

## Using the dataset with the code

The inference harness reads images from local paths (each record's `image_path`, e.g.
`./images/gpt_image_gen/textual/...png`), so `multibbq download` bridges the Hub to those
paths:

```bash
pip install "multibbq[hf]"
multibbq download                      # pull images + aux archives into ./images/
multibbq download --no-perturbations   # skip the ~16 GB perturbation sets
multibbq download --root /data         # write the images/ tree under /data
```

Other flags: `--no-primary` (skip the gpt_image_gen/ and imagen4ultra_image_gen/ trees),
`--no-realworld`, `--no-blank`, and `--repo` (defaults to `MLL-Lab/MultiBBQ`). It writes
each config's images to the harness paths and lays out the auxiliary and perturbation
images, so a clone plus `multibbq download` is ready to run.

## Building and pushing the dataset

Maintainers rebuild the dataset from a local snapshot of images + metadata with
[`../scripts/build_hf_dataset.py`](../scripts/build_hf_dataset.py):

```bash
pip install "multibbq[hf]"
huggingface-cli login                  # or set HF_TOKEN
python scripts/build_hf_dataset.py --source <snapshot> --stage-dir hf_stage        # build + verify
python scripts/build_hf_dataset.py --source <snapshot> --stage-dir hf_stage --push # + upload
```

The script embeds each image into sharded parquet, assembles `hf_stage/` (four configs +
`image_archives/` + the card), uploads that tree to `MLL-Lab/MultiBBQ` with `upload_folder`,
and uploads the perturbation sets to `MLL-Lab/MultiBBQ-perturbations`. It deliberately avoids
`Dataset.push_to_hub` (whose card-merge step can raise a spurious `UnicodeDecodeError`). The
card's YAML front matter declares the license, tags, and config layout for the viewer.

## The outputs repo

The raw model outputs (`results/`) and computed metrics (`analysis/`) are reproduction
artifacts, not a dataset to train on, so they live in a separate HF dataset repo. Upload with
`upload_folder` (or the CLI):

```bash
huggingface-cli upload MLL-Lab/MultiBBQ-results results/  results/  --repo-type dataset
huggingface-cli upload MLL-Lab/MultiBBQ-results analysis/ analysis/ --repo-type dataset
```

Everything in `analysis/` is regenerable from `results/` with `multibbq pipeline`, so the
outputs repo is a convenience and an audit trail. See [RESULTS.md](RESULTS.md) and the repo
card [`hf_results_card.md`](hf_results_card.md).
