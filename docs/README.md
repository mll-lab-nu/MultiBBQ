# Documentation

Four folders, ordered the way you are likely to need them. If you only read one page,
read the [project README](../README.md) first: it has the Quick Start.

## Pick your path

**"I want to run an evaluation"** (30 minutes)
1. [getting-started/installation.md](getting-started/installation.md): conda env, `pip install -e .`, API keys, image download.
2. [getting-started/running.md](getting-started/running.md): the `run -> score -> combine -> aggregate -> pipeline` chain, flags, output layout, cost.

**"I want to understand the benchmark"**
1. [benchmark/dataset.md](benchmark/dataset.md): what a record looks like (schema, image manifest).
2. [benchmark/metrics.md](benchmark/metrics.md): Fairness / Bias / Unknown-rate, formulas, anti-gaming.
3. [benchmark/experiments.md](benchmark/experiments.md): the 11 evaluation settings mapped to paper sections.
4. [benchmark/dataset-construction.md](benchmark/dataset-construction.md): how the dataset and images were built.

**"I want to reproduce the paper"**
1. [benchmark/reproducing.md](benchmark/reproducing.md): step-by-step, released-images vs regenerate-images paths.
2. [benchmark/models.md](benchmark/models.md): the 28 model ids and download links.
3. [benchmark/RESULTS.md](benchmark/RESULTS.md): where the image set, raw results, and computed analysis live.

**"I want to evaluate my own model"**
1. [extending/evaluate-your-own-model.md](extending/evaluate-your-own-model.md): add a new model / adapter.
2. [extending/llm-evaluation.md](extending/llm-evaluation.md): evaluating text-only LLMs.
3. [extending/extending.md](extending/extending.md): add an experiment, a metric, or a bias category.

**"I want the data itself"**
1. [getting-started/installation.md Dataset & images section](getting-started/installation.md#dataset--images): the download guide - all methods, sizes, and where files land.
2. [huggingface/hf.md](huggingface/hf.md): the HuggingFace repo layout, self-serve extraction, and the maintainer build/upload flow.
3. The Hub cards, as published: [dataset](huggingface/hf_dataset_card.md), [realworld](huggingface/hf_realworld_card.md), [perturbations](huggingface/hf_perturbations_card.md), [results](huggingface/hf_results_card.md).

## Folder map

| Folder | Contents |
|---|---|
| [getting-started/](getting-started/) | Install and run: `installation.md`, `running.md`. |
| [benchmark/](benchmark/) | The study itself: `dataset.md`, `dataset-construction.md`, `metrics.md`, `experiments.md`, `models.md`, `reproducing.md`, `RESULTS.md`. |
| [extending/](extending/) | Use it on your own models: `evaluate-your-own-model.md`, `llm-evaluation.md`, `extending.md`. |
| [huggingface/](huggingface/) | The released artifacts: `hf.md` (layout + download), plus the three Hub dataset cards. |
