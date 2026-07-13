# Data

Everything the benchmark reads or was built from, one subfolder per role. Metadata and
construction materials are committed; the **full image set is downloaded** (see the
[download guide](../docs/getting-started/installation.md#dataset--images)).

| Subfolder | Role | In git? |
|---|---|---|
| [`metadata/`](metadata/) | The evaluation record tables the harness reads | yes |
| [`images/`](images/) | The image trees, laid out by `multibbq download` | only the blank canvas |
| [`images_sample/`](images_sample/) | A small browsable preview of the images | yes |
| [`construction/`](construction/) | Templates, vocabulary, and tables used to build the dataset | yes |

## `metadata/` - evaluation record tables

Grouped by image generator, one file per modality; `.csv` is for human inspection,
`.json` is what the inference harness loads (from `./data/metadata/<generator>/`).

| Path | Contents |
|---|---|
| `metadata/gpt_image_gen/multibbq_visual_language.{csv,json}` | GPT-Image-1 images, visual-language eval |
| `metadata/gpt_image_gen/multibbq_visual_only.{csv,json}` | GPT-Image-1 images, visual-only eval |
| `metadata/imagen4ultra_image_gen/multibbq_visual_language.{csv,json}` | Imagen-4-Ultra images, visual-language eval |
| `metadata/imagen4ultra_image_gen/multibbq_visual_only.{csv,json}` | Imagen-4-Ultra images, visual-only eval |

Each row carries both **masked** (demographic-neutral, e.g. "the person on the left")
and **unmasked** context/option fields, the question pair (negative / non-negative),
the correct-answer and stereotype/non-stereotype indices, and the `image_path` the
harness expects under `./data/images/`. The non-image fields are identical across the
two generators; only `image_path` differs.

Each table has 410 rows. In the two visual-only tables, 2 rows per generator reference
images the generator refused to produce, so visual-only evaluations cover 408 items
(the harness prints `Image not found` and skips those rows).

## `images/` - the image trees

Laid out by `multibbq download` (main / `--realworld` / `--perturbations`); mirrors
`metadata/` by generator, e.g. `images/gpt_image_gen/{visual_language,visual_only}/`.
Only the blank canvas (`pure_white_1024_1024.png`, the `unmasked_wo_img` control) is
tracked in git.

## `construction/` - how the dataset was built

Inputs and intermediate tables of the provenance notebooks (not used at evaluation time):

| File | Role |
|---|---|
| `templates/` | BBQ-derived context/question templates + demographic vocabulary (see its [README](construction/templates/README.md)). |
| `multibbq_template_table.csv` | The slotted record table produced by `notebooks/gen_template.ipynb`; every downstream step reads it. |
| `multibbq_template_table_w_face_id.csv` | Same, with per-item real-face ids added by `notebooks/gen_realworld.ipynb`. |
| `real_world_images.csv` | The real-face catalog used to assemble the real-world split. |

**Dataset license: CC-BY-4.0.**
