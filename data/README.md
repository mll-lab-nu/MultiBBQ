# Data

The MultiBBQ dataset. Metadata is committed here; the **full image set is released
separately on HuggingFace** (see [`../docs/RESULTS.md`](../docs/RESULTS.md)); only the
small [`images_sample/`](images_sample/) preview is in git.

**Evaluation sets**, grouped by image generator, one file per modality. `.csv` is for
human inspection, `.json` is what the inference harness loads.

| Path | Contents |
|---|---|
| `gpt_image_gen/mmbbq_visual_language.{csv,json}` | GPT-Image-1 images, visual-language eval |
| `gpt_image_gen/mmbbq_visual_only.{csv,json}` | GPT-Image-1 images, visual-only eval |
| `imagen4ultra_image_gen/mmbbq_visual_language.{csv,json}` | Imagen-4-Ultra images, visual-language eval |
| `imagen4ultra_image_gen/mmbbq_visual_only.{csv,json}` | Imagen-4-Ultra images, visual-only eval |

Each row carries both **masked** (demographic-neutral, e.g. "the person on the left")
and **unmasked** context/option fields, the question pair (negative / non-negative),
the correct-answer and stereotype/non-stereotype indices, and the `image_path` the
harness expects under `./images/`.

**Construction inputs** (used by the notebooks, not by evaluation):

| File | Role |
|---|---|
| `mmbbq_temp_revised.csv` | Revised template table consumed by `notebooks/gen_template.ipynb`. |
| `mmbbq_temp_revised_w_face_id.csv` | Same, with per-image face ids for real-world matching. |
| `real_world_images.csv` | Mapping for the real-world image split (`notebooks/gen_realworld.ipynb`). |

Template/vocabulary sources live in [`../templates/`](../templates/). **Dataset license:
CC-BY-4.0.**
