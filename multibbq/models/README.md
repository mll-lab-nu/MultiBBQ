# multibbq.models: model wrappers

One wrapper per model family behind a single `ModelFactory`. Each wrapper exposes a
uniform `run(image_path, user_prompt, system_msg)`; the three historical per-variant files
(`_reasoning` / `_temp` / `_quant`) are collapsed into **three orthogonal constructor
axes** on `BaseModel`:

- **`mode`**: `'default'` | `'reasoning'` | `'temp'` (sets `gen_kwargs`; `reasoning` also
  injects a think-rule into the prompt/system message).
- **`quant`**: load the checkpoint quantized (per-family precision: blip2/internvl =
  8-bit, llava = 4-bit).
- **`temperature`**: sampled decoding for `mode='temp'`.

| File | Role |
|---|---|
| `base.py` | `BaseModel`: the three-axis base class (`gen_kwargs`, `think_rule`, uniform `run` contract). |
| `factory.py` | `ModelFactory`: parses a model id → family, dispatches to the wrapper with `mode`/`quant`/`temperature` (and `text_only`). Lazy imports so one model's deps don't pull in the rest. |
| `qwen_vl.py` | `Qwen2_5VLModel` (3B / 7B / 32B / 72B). |
| `internvl.py` | `InternVL3_5Model` (1B / 2B / 4B / 8B / 14B / 38B), incl. dynamic-tiling preprocessing. |
| `llava.py` | `LlavaModel` (34B / Mistral-7B / Vicuna-13B). |
| `gemma.py` | `Gemma3Model` (4B / 12B / 27B IT). |
| `minicpm.py` | `MiniCPMVModel` (4.5). |
| `deepseek_vl.py` | `DeepSeekVLModel` (1.3B / 7B); needs the `deepseek_vl` vendor package. |
| `blip2.py` | `Blip2Model` (OPT-2.7B / 6.7B). |
| `fuyu.py` | `FuyuModel` (8B). |
| `gpt.py` | `GPT4OModel`, `GPT5Model` (base / mini / nano); OpenAI API. |
| `gemini.py` | `GeminiModel` (2.5 flash / flash-lite); Vertex AI. |
| `text.py` | `HFTextModel`: **text-only** HF causal LM (no image) for the `llm` experiment, the language-only evaluation path. See [../../docs/llm-evaluation.md](../../docs/llm-evaluation.md). |

**Coverage:** every vision family has `default` + `reasoning`; all local families add
`temp`; `quant` exists for blip2 / internvl / llava (Qwen reuses the base class). API
families (gpt / gemini) are `default` + `reasoning` only. `HFTextModel` covers any HF chat
LLM. Full id list: [../../docs/models.md](../../docs/models.md).

**Adding a model:** write a wrapper subclassing `BaseModel` with a `run(...)`, then add one
row to `_REGISTRY` in `factory.py`. See [../../docs/evaluate-your-own-model.md](../../docs/evaluate-your-own-model.md).
