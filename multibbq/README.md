# multibbq: the package

The MultiBBQ Python package: one CLI, one inference loop, one wrapper per model
family, and the scoring subpackage. Installed as `multibbq` (see the top-level
[`pyproject.toml`](../pyproject.toml)); run as `multibbq <subcommand>` or
`python -m multibbq`.

| File | Role |
|---|---|
| `cli.py` | CLI entry point; defines the `run` / `download` / `score` / `combine` / `aggregate` / `pipeline` subcommands. Model and HF imports are lazy so the metric subcommands work without a GPU or HF stack. |
| `inference.py` | The unified prediction loop: image resolution, field selection, prompt building, model call, result serialization. One loop for every experiment. |
| `experiments.py` | The `EXPERIMENTS` table (11 settings) and the `SYSTEM_MSGS` instruction set: the single source of truth for what each `--experiment` does. |
| `hf.py` | `multibbq download`: extract the main image set from the HuggingFace parquet configs and snapshot the realworld / perturbation repos, laying out `./images/` for the harness. See [`../docs/huggingface/hf.md`](../docs/huggingface/hf.md). |
| `utils.py` | Inference-path text utility (`add_in_the_image`, used by `context_unmasked`). |
| `models/` | One wrapper per model family + the unified `ModelFactory`. See [`models/README.md`](models/README.md). |
| `metrics/` | Fairness / Bias / Unknown-rate scoring and aggregation. See [`metrics/README.md`](metrics/README.md). |
| `__main__.py` | Enables `python -m multibbq`. |

Entry chain: `multibbq run …` → `cli.cmd_run` → `inference.run` →
`models.factory.ModelFactory` → the resolved wrapper.
