# Running evaluations

The `multibbq` CLI is a five-verb pipeline. Inference produces raw generations; the metric
verbs turn them into scores.

```
run ──▶ results/…json ──▶ score ──▶ *_w_metrics.json ──▶ combine ──▶ combined_metrics.json ──▶ aggregate ──▶ CSVs + FS/BS totals
                                     └────────────────────── pipeline (score → combine → aggregate) ──────────────────────┘
```

`run` imports the model stack lazily; `score`/`combine`/`aggregate`/`pipeline` need only
pandas.

## `multibbq run`: inference

```bash
multibbq run <model_id> --experiment <name> [flags]
```

`<model_id>` is a HuggingFace or API id (e.g. `OpenGVLab/InternVL3_5-8B`, `gpt-5-mini`,
`Qwen/Qwen2.5-7B-Instruct` for text-only).

**Common flags**

| Flag | Values | Meaning |
|---|---|---|
| `--experiment` | see [experiments.md](../benchmark/experiments.md) | evaluation setting (default `main`) |
| `--data_id` | `gpt_image_gen` \| `imagen4ultra_image_gen` | which generator's images/data |
| `--textual_context` | `true` \| `false` | visual-language vs. visual-only |
| `--ambiguous` | `true` \| `false` | ambiguous vs. disambiguous context |
| `--negative` | `true` \| `false` | negative vs. non-negative question |

**Experiment-specific flags**

| Flag | Used by | Values |
|---|---|---|
| `--img_aug_type` | `aug_img`, `img_label` | `noise` \| `brightness_up` \| `brightness_down` \| `contrast_up` \| `contrast_down` \| `compression` \| `resize_l` \| `resize_s` \| `label` (required by `img_label`) \| plus the `brightness` / `contrast` baselines |
| `--temperature` | `temp` | `0.2` \| `0.4` \| `0.6` \| `0.8` \| `1.0` |
| `--reasoning_mode` | `reasoning` | `nonreasoning` \| `reasoning` \| `nonreasoning_w_fairness` \| `reasoning_w_fairness` |

**Output layout**

```
results/<data_id>_<token>/<model_id>/<model>_<size>_<modality>_<qtype>_<ambiguity>[suffix].json
results/<data_id>_<token>/<model_id>_logs/….log
```

`<token>` is the experiment's subdirectory (`main`, `quant`, `temp_0.6`, `noise`,
`realworld`, `reasoning`, `llm`, …). `<modality>` is `visual_language` / `visual_only`, or
`text` for the `llm` experiment. Each JSON is `{"data": [ … ]}`, one record per example
(`image`, `category`, `options`, `pred`, `correct_option_idx`, and the stereotype / unknown
indices), the exact fields the scorer reads.

## Metric verbs

| Verb | In → out |
|---|---|
| `score` | one result file → scores on stdout; or a directory → mirrored `*_w_metrics.json` tree |
| `combine` | a `*_w_metrics.json` tree → one `combined_metrics.json` |
| `aggregate` | `combined_metrics.json` → per-category CSVs + `FS_total` / `BS_total` |
| `pipeline` | a results directory → all of the above in one shot |

```bash
multibbq score    --input results/gpt_image_gen_main/openai/gpt-5/gpt_5_visual_language_negative_ambiguous.json
multibbq pipeline --input results/gpt_image_gen_main --output analysis/gpt_image_gen_main
```

Useful flags: `--score {all,fairness,bias,unk}` (which fields to print), `--tail-slice N`
(scan only the last `N` chars for the Unknown option, for long reasoning outputs),
`--skip-existing` (resume a partially-scored directory). Definitions: [metrics.md](../benchmark/metrics.md).

## Batch launchers

`scripts/` has one Slurm/bash template per experiment (see [`../../scripts/README.md`](../../scripts/README.md)).
Edit the resource header + model list, then:

```bash
sbatch scripts/eval_main.sh        # cluster
bash   scripts/eval_main_cpu.sh    # local, no Slurm
```

## Cost & compute notes

- **Open-source models** run on GPU via HuggingFace Transformers; memory scales with size
  (1B–72B). The paper used H100s with **greedy decoding** (except the `temp` study).
- **API models (GPT / Gemini)** cost scales with request count: roughly
  410 examples × up to 6 context-by-question conditions per model (~2,460 requests).
  Estimate the bill before launching a full 28-model sweep; start with one small
  condition to calibrate.
- **Resume:** `--skip-existing` (score) and the per-condition output files make runs
  restartable, so re-running skips completed files.
