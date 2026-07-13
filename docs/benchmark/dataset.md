# Dataset

MultiBBQ extends the language-only **BBQ** benchmark to the multimodal setting, pairing
attested social biases with AI-generated photorealistic images. This page is the dataset
card + record schema. How it was built: [dataset-construction.md](dataset-construction.md).

## Card

- **Task:** multiple-choice social-bias QA over an image (or text-only, for LLM eval).
- **Categories (4):** Race, Gender, Religion, Age, chosen for visually identifiable cues.
- **Size:** **410 examples / 2,460 QA pairs.**
- **License:** CC-BY-4.0. **Images:** released on HuggingFace (see [RESULTS.md](RESULTS.md)).

| Category | Examples | QA pairs | Subgroups |
|---|---|---|---|
| Race | 127 | 762 | Black / White / East Asian / Native American / Hispanic / Arab |
| Gender | 50 | 300 | Male / Female |
| Religion | 134 | 804 | Christian / Hindu / Muslim / Buddhist / Jewish / Orthodox / Atheist |
| Age | 99 | 594 | Young / Middle-aged / Old Adult |

**Conditions per example (6):** three contexts (Visual-Only Ambiguous, Visual-Language
Ambiguous, Visual-Language Disambiguated, with no VO-Disambiguated, as synthetic images are
intrinsically ambiguous) × two question polarities (negative / non-negative). Three
answer choices: "The person on the left", "The person on the right", "Unknown".

## Files

Grouped by image generator; `.csv` for inspection, `.json` for inference (see
[`../../data/README.md`](../../data/README.md)):

```
data/{gpt_image_gen,imagen4ultra_image_gen}/multibbq_{visual_language,visual_only}.{csv,json}
```

Non-image fields are identical across generators; only `image_path` differs.

## Record schema

Each JSON record (one example, one modality):

| Field | Meaning |
|---|---|
| `category`, `q_id`, `c_id` | bias category and question / context ids |
| `ambig_context` / `ambig_context_masked` | ambiguous context: **unmasked** (real demographic terms) vs **masked** ("the person on the left/right") |
| `disambig_context` / `disambig_context_masked` | the disambiguating sentence, unmasked / masked |
| `neg_q`, `nonneg_q` | the negative and non-negative questions |
| `options` / `options_masked` | the three answer choices, unmasked / masked |
| `neg_label_idx` / `neg_label_name` | gold answer for the negative question (disambiguated) |
| `nonneg_label_idx` / `nonneg_label_name` | gold answer for the non-negative question |
| `unk_label_idx` | index of the "Unknown" option (position is randomized per item) |
| `stereotype_group_idx` / `stereotype_group_name` | the socially-stereotyped option/group |
| `nonstereotype_group_idx` / `nonstereotype_group_name` | the non-stereotyped option/group |
| `stereotypes` / `nonstereotypes` | subgroup lists behind each side |
| `name1`, `name2`, `person_on_the_left`, `person_on_the_right` | the two entities and their layout |
| `word1`, `word2` | template slot fillers |
| `image_path` | path under `./data/images/…` that inference reads |
| `visual_only_ambig_prompt_w_position` / `_wo_position`, `visual_textual_prompt` | the image-generation prompts (provenance) |

**Masking** is MultiBBQ's shortcut-reasoning control: in *ambiguous* contexts and options,
demographic terms are replaced by positional references so a model cannot infer "Unknown"
from language alone; in *disambiguated* contexts the demographic terms are retained in the
added sentence (the positional gold answer still requires reading the image). Answer-option
order and stereotype/non-stereotype assignment are randomized to remove order bias.

## Which fields each experiment uses

- **MLLM experiments** (`main`, `quant`, …) use the **masked** context/options + the image.
- **Backbone / unmasked** (`unmasked_w_img`, `unmasked_wo_img`, `context_unmasked`) use the
  **unmasked** text.
- **Text-only LLM** (`llm`) uses the unmasked text with no image (the "in the image"
  phrasing is stripped). See [llm-evaluation.md](../extending/llm-evaluation.md).

## Loading

```python
import json
data = json.load(open("data/gpt_image_gen/multibbq_visual_language.json"))
print(len(data), data[0]["category"], data[0]["options"])
```

The metrics are index-based over `pred`, `correct_option_idx`, and the
stereotype/unknown indices, so scoring is modality-agnostic ([metrics.md](metrics.md)).
