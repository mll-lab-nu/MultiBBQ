# Extending MultiBBQ

MultiBBQ's evaluation surface is data-driven: experiments, metrics, and bias
categories are all table/data entries, not code paths. This document is the
reference for the three extension points beyond adding a model (for that, see
[evaluate-your-own-model.md](evaluate-your-own-model.md)), each in **Where / The
contract / Validate** form.

| # | Extension point | Primary file(s) | Selected by |
|---|---|---|---|
| 1 | **New experiment** | [../../multibbq/experiments.py](../../multibbq/experiments.py) | `--experiment` |
| 2 | **New metric field** | [../../multibbq/metrics/scorer.py](../../multibbq/metrics/scorer.py) + [../../multibbq/cli.py](../../multibbq/cli.py) | `--score` |
| 3 | **New bias category** | the data (`category` field) + `--categories` | data + CLI flag |

---

## 1. Add a new experiment

### Where

The single `EXPERIMENTS` dict in
[../../multibbq/experiments.py](../../multibbq/experiments.py). Each key is one
`--experiment` value; the value is a dict of orthogonal axes plus a results
`token`. The unified inference loop in
[../../multibbq/inference.py](../../multibbq/inference.py) reads these keys, so there is no
per-experiment script to add.

### The contract

Add one row. Every axis key:

| Key | Values | Meaning |
|---|---|---|
| `mode` | `default` \| `temp` | Model wrapper mode (`reasoning` is resolved at runtime from `--reasoning_mode`, so use `default` + `retry=True` for reasoning experiments). |
| `quant` | `bool` | Load the model quantized. |
| `fields` | `masked` \| `unmasked` | `masked` uses the demographic-neutral `*_masked` text columns (MultiBBQ's language-leakage control); `unmasked` uses the raw BBQ columns. |
| `image` | `dataset` \| `aug` \| `blank` \| `realworld` \| `none` | Image source: the generated image, a perturbed variant, a white canvas, a real-world photo, or no image (text-only). |
| `options` | `plain` \| `label` | `plain` shows the (masked) choices; `label` replaces the non-answer options with `person A/B/C`. |
| `disambig` | `plain` \| `masked` | Which disambiguating-context column to use (`img_label` uses `disambig_context_masked`). |
| `inject` | `bool` | Re-inject demographic names into the ambiguous context (the `context_unmasked` leakage control). |
| `vo` | `bool` | Whether the visual-only split is supported for this setting. |
| `retry` | `bool` | Wrap `model.run` in the 3-attempt retry loop (needed for long reasoning outputs that intermittently return `None`). |
| `token` | str | Results subdirectory: `results/<data_id>_<token>/`. Overridden at runtime for some experiments: `image="aug"` uses `--img_aug_type`, `temp` uses `temp_<temperature>`, and `reasoning` uses `--reasoning_mode`. |
| `text_only` | `bool` (optional) | Route through the factory's `text_only=True` path (`HFTextModel`, no vision wrapper). Used by `llm`. |
| `strip_image_ref` | `bool` (optional) | Strip `" in the image"` from the question (for text-only eval where there is no image). Used by `llm`. |

Example (a new "grayscale image" study reusing the baseline text setup):

```python
EXPERIMENTS = {
    ...
    "gray_img": dict(mode="default", quant=False, fields="masked", image="aug",
                     options="plain", disambig="plain", inject=False, vo=True,
                     retry=False, token="aug"),
}
```

Because `image="aug"`, the harness resolves both the image tree and the results
subdirectory from `--img_aug_type` (the registry `token` is overridden at runtime):
place your grayscale copies under `data/images/gpt_image_gen_gray/`, mirroring the
main set's layout, and add `"gray"` to the `--img_aug_type` choices in the `run`
parser. The `--experiment` flag itself needs no CLI edit:
`run`'s parser declares `--experiment choices=sorted(EXPERIMENTS)` in
[../../multibbq/cli.py](../../multibbq/cli.py), so a new key becomes a valid value
automatically. If your experiment needs a brand-new per-run knob, add the `argparse`
argument to the `run` parser and read it in `inference.py`.

### Validate

```bash
multibbq run org/MyModel-7B --experiment gray_img --data_id gpt_image_gen --img_aug_type gray
```

Confirm results land under `results/gpt_image_gen_gray/…` and that scoring succeeds
([running.md](../getting-started/running.md), [metrics.md](../benchmark/metrics.md)). Cross-check against
[experiments.md](../benchmark/experiments.md), which documents each shipped row.

---

## 2. Add a new metric

### Where

Scoring lives in [../../multibbq/metrics/scorer.py](../../multibbq/metrics/scorer.py).
Both entry points (`eval_visual_only(data, neg, tail_slice=None)` and
`eval_visual_language(data, ambig, neg, tail_slice=None)`) return the same schema:

```python
{
    "overall":     {"fairness_score", "bias_score", "unk_rate"},
    "by_category": {<category>: {"fairness_score", "bias_score", "unk_rate"}, ...},
}
```

### The contract

To add a field (say `err_rate`, the -1/unparseable rate):

1. **Compute it in both scorers.** Accumulate the per-row count in the
   `category` loop, then emit it in the per-category `category_metrics[category]`
   dict *and* in the `overall` dict, mirroring how `fairness_score` /
   `bias_score` / `unk_rate` are computed with `_safe_div`. Keep both scorers in
   sync; they share the schema.
2. **Surface it via the CLI.** `--score` in the `score` subparser
   ([../../multibbq/cli.py](../../multibbq/cli.py)) is `choices=["all", "fairness",
   "bias", "unk"]`, resolved by `_resolve_score_flag` against the mapping
   `{"fairness": "fairness_score", "bias": "bias_score", "unk": "unk_rate"}`.
   Add your key to `_SCORE_KEYS`, add a `choices` entry, and add its mapping
   row. `_filter_scores` then prunes to the selected key(s) for both `overall`
   and `by_category` automatically.
3. **(If it should reach the CSVs)** the aggregate layer
   ([../../multibbq/metrics/aggregate.py](../../multibbq/metrics/aggregate.py)) reads
   only `fairness_score` / `bias_score` into its BBQ-shaped MultiIndex CSVs. A
   new field surfaces through `score`/`combine` but will not appear in the
   aggregated CSVs unless you extend those builders too.

### Validate

Score a reference file and diff the new field against a hand-computed value:

```bash
multibbq score -i results/gpt_image_gen_main/org/MyModel-7B/<one>.json -o /tmp/ref_w_metrics.json --score all
```

Check the printed `overall` / `by_category` block includes your field with the
expected value.

---

## 3. Add a new bias category

### Where

Categories are pure data; nothing in the scorer enumerates them. Each result row
carries a `category` string; the scorers group on `row["category"]`
(`category_data[row["category"]].append(row)`), so a new category value appears in
`by_category` automatically the moment rows carry it.

### The contract

1. **Add data rows** with your new `category` value (e.g. `"disability"`),
   alongside the existing fairness fields each row needs (`pred`,
   `stereotype_group_idx`, `nonstereotype_group_idx`, `unk_label_idx`, and
   `correct_option_idx` for visual-language). The scorer's per-category grouping
   requires no code change.
2. **Pass it to aggregate.** The aggregate/pipeline steps take an explicit
   `--categories` list (default `["gender", "race", "religion", "age"]` in
   `_add_metric_common`, [../../multibbq/cli.py](../../multibbq/cli.py)). Category CSVs
   and the `category_total_scores_summary.csv` only include categories you name:

   ```bash
   multibbq pipeline -i results/... -o out/ \
       --categories gender race religion age disability
   ```

   `create_all_category_metrics_csvs` writes one CSV per category found in the
   data, but `cal_total_scores` / `combine_category_totals` only process the
   `--categories` you pass (and `overall`). If you also want it in the
   per-main-category average CSV, add its title-cased form to
   `--subcategory-order`.

### Validate

```bash
multibbq score -i results/... -o out/analysis          # per-category scores appear
multibbq aggregate -i out/combined_metrics.json \
    --csv-dir out/csv --metrics-dir out/metrics \
    --categories gender race religion age disability
```

Confirm a `disability_metrics_summary_sorted.csv` is written and a `disability`
column appears in `category_total_scores_summary.csv`. See [metrics.md](../benchmark/metrics.md)
for the full scoring → combine → aggregate pipeline.
