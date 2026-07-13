# Evaluating text-only LLMs

MultiBBQ is built on **BBQ**, a language-only bias benchmark, so every record carries the
full **unmasked** text: context, question, and options with the real demographic terms.
That makes MultiBBQ a *superset* that can also evaluate **text-only LLMs** (Llama, Qwen,
Gemma, Mistral, DeepSeek, …), not just multimodal ones. The metrics are index-based and
**modality-agnostic**, so scoring is identical.

## Run it

```bash
multibbq run "Qwen/Qwen2.5-7B-Instruct" --experiment llm \
    --ambiguous true --negative true
```

The `llm` experiment:

- uses **no image** (any HuggingFace causal LM, loaded via
  [`../../multibbq/models/text.py`](../../multibbq/models/text.py));
- feeds the **unmasked** BBQ-style context + question + options;
- strips `" in the image"` from the question to recover clean BBQ phrasing
  (398/410 questions contain it);
- forces visual-language mode (the text context *is* the task, since a "visual-only" text run
  would be a degenerate question-only prompt).

Outputs land in `results/<data_id>_llm/<model_id>/…_text_…json` and score exactly like any
other run:

```bash
multibbq pipeline --input results/gpt_image_gen_llm --output analysis/gpt_image_gen_llm
```

## How the metric adapts

Nothing in the **scoring math** changes: an LLM's A/B/C answer maps to an option index and
flows through the same Fairness / Bias / Unknown-rate scorer. Two things are handled so the
result is *honest*, not merely functional:

1. **Modality label.** Text-only outputs are tagged `text` (not `visual_language`), so they
   are never conflated with a multimodal run.
2. **The Total.** `FS_total` / `BS_total` are normally a **3-scenario** harmonic mean
   (Visual-Only Am + Visual-Language Am + Visual-Language Dis). A text-only LLM has **no
   visual-only scenario**, so its Total reduces to a **2-scenario harmonic mean over the
   language scenarios (Am + Dis)**, so the Visual-Only columns stay empty and the harmonic
   mean uses two terms. Per-scenario FS/BS are directly comparable to an MLLM's `VL.Am` /
   `VL.Dis`; the 2-scenario Total is **not** directly comparable to an MLLM's 3-scenario
   Total. See [metrics.md](../benchmark/metrics.md).

## Relationship to the backbone experiments

The paper measures the *language side* of an MLLM's bias with the **backbone** experiments
(`unmasked_wo_img`: unmasked text + a blank white canvas, run through the MLLM interface).
The `llm` experiment is the natural complement: it evaluates a **standalone** text LLM with
no vision stack at all, for example to compare an MLLM against its own backbone LLM, or to place
a language-only model on the same fairness axes.

| | image | text | model interface |
|---|---|---|---|
| `unmasked_w_img` | real image | unmasked | MLLM |
| `unmasked_wo_img` | blank canvas | unmasked | MLLM |
| `llm` | none | unmasked | **text LLM** |

## Supported models & extension

`HFTextModel` is generic across HF chat LLMs (`AutoModelForCausalLM` + the tokenizer's chat
template) and honors the three axes (`mode='reasoning'`, `mode='temp'`, `quant`). To add an
API text model or a custom endpoint, subclass `BaseModel` the same way (see
[evaluate-your-own-model.md](evaluate-your-own-model.md)).
