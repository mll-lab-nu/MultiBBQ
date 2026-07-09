# Scripts

Batch launchers, one per experiment. Each is a **Slurm/bash template** that loops a
model list through `multibbq run --experiment …`; edit the account / partition / gres /
time / mem header and the model list before submitting. Run from the repository root.

| Script | Experiment(s) | Notes |
|---|---|---|
| `eval_main.sh` / `eval_main_cpu.sh` | `main` | Baseline; the `_cpu` variant runs locally without Slurm. |
| `eval_quantization.sh` | `quant` | Quantized backbones. |
| `eval_temp.sh` | `temp` | Sweeps `--temperature` over {0.2 … 1.0}. |
| `eval_reasoning.sh` / `eval_reasoning_cpu.sh` | `reasoning` | Sweeps `--reasoning_mode`. |
| `eval_aug_img.sh` | `aug_img` | Sweeps `--img_aug_type` (noise / brightness / …). |
| `eval_main_label.sh` | `img_label` | Generic-label options. |
| `eval_realworld.sh` / `eval_realworld_cpu.sh` | `realworld` | Real-world images (visual-language only). |
| `eval_main_context_unmasked.sh` / `_cpu` | `context_unmasked` | Demographic names injected into context. |
| `eval_unmasked_w_img.sh` / `eval_unmasked_wo_img.sh` | `unmasked_w_img` / `unmasked_wo_img` | Backbone (unmasked-text) studies. |
| `download_models.sh` | (none) | Pre-fetch open-source checkpoints. |

Usual path: edit a script → `sbatch scripts/eval_main.sh` (or `bash scripts/eval_main_cpu.sh`
locally). See [`../docs/running.md`](../docs/running.md).
