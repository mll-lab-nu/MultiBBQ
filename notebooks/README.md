# Notebooks

Provenance notebooks for **dataset construction** and **image generation**. They are
**not** part of the evaluation path (use `multibbq run` for inference and
`multibbq score` / `pipeline` for metrics). They document how the released dataset and
images were produced.

| Notebook | Purpose | Inputs |
|---|---|---|
| `gen_template.ipynb` | Slot vocabulary into the category templates to build the MultiBBQ question/context table (writes `../data/construction/multibbq_template_table.csv`). | [`../data/construction/templates/`](../data/construction/templates/), `utils.py` (here) |
| `gen_images_gpt_image_gen.ipynb` | Synthesize images with GPT-Image-1. | the template table above |
| `gen_images_imagen4ultra_image_gen.ipynb` | Synthesize images with Imagen-4-Ultra. | the template table above |
| `gen_realworld.ipynb` | Assemble the real-world image split and its metadata (writes `../data/construction/multibbq_template_table_w_face_id.csv`). | `../data/construction/real_world_images.csv`, `../data/construction/multibbq_template_table.csv` |
| `utils.py` | Dataset-generation helpers (`return_list_from_string`, slotting, template dicts) used by `gen_template.ipynb`. |

All notebooks run with this folder as the working directory: they read `../data/construction/templates/`
and `../data/`, and the image notebooks write to `../data/images/`, which is the
repository `data/images/` tree the harness reads from. No manual file moves are needed.

> Image generation is **non-deterministic** (and the generator APIs evolve), so a rerun
> will not reproduce the released images byte-for-byte. To reproduce the paper, prefer
> `multibbq download` (the released image set). See
> [`../docs/benchmark/dataset-construction.md`](../docs/benchmark/dataset-construction.md).
