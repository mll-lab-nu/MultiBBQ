# multibbq.metrics: fairness scoring

Computes MultiBBQ's **Fairness Score**, **Bias Score**, and **Unknown-rate** from raw
inference outputs, and aggregates many models into the paper's comparison tables. Pure
pandas, no GPU stack. Exposed both as a Python API and via the `multibbq`
`score` / `combine` / `aggregate` / `pipeline` subcommands.

| File | Role |
|---|---|
| `parsers.py` | `parse_pred`: map a model's raw text answer to an option index, with unknown-synonym matching (`--tail-slice` for long reasoning outputs). |
| `scorer.py` | `eval_visual_only` / `eval_visual_language`: per-category and overall Fairness / Bias / Unknown-rate for one result set. |
| `io.py` | `eval_file` / `eval_directory` / `combine_metrics`: score files (filename → flags), mirror a results tree, and merge into `combined_metrics.json`. |
| `aggregate.py` | `combined_metrics.json` → per-category CSV summaries and `FS_total` / `BS_total` via a 3-term harmonic mean; the `pipeline` back-end. |

Python API:

```python
from multibbq.metrics import eval_file

metrics = eval_file("model_visual_language_negative_ambiguous.json")
# {"overall": {"fairness_score", "bias_score", "unk_rate"}, "by_category": {...}}
```

Filename convention used for auto-inference:
`*_{visual_only,visual_language}_{negative,nonnegative}_{ambiguous,disambiguous}.json`.

Metric definitions: [`../../docs/benchmark/metrics.md`](../../docs/benchmark/metrics.md).
