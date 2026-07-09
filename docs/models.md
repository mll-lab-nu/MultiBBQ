# Supported models

**28 models across 11 families.** Pass any of these ids as `multibbq run <model_id>`.
Open-source checkpoints auto-download from HuggingFace; API models read credentials from
the environment (see [installation.md](installation.md)).

## Closed-source (API)

| Family | Variants | id(s) | Credentials |
|---|---|---|---|
| GPT-4o | (none) | `gpt-4o` | `OPENAI_API_KEY` |
| GPT-5 | base / mini / nano | `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | `OPENAI_API_KEY` |
| Gemini 2.5 | flash / flash-lite | `gemini-2.5-flash`, `gemini-2.5-flash-lite` | `GOOGLE_CLOUD_PROJECT` |

## Open-source (HuggingFace)

| Family | Variants | Example id |
|---|---|---|
| Fuyu | 8B | `adept/fuyu-8b` |
| DeepSeek-VL | 1.3B / 7B | `deepseek-ai/deepseek-vl-7b-chat` |
| Gemma3-IT | 4B / 12B / 27B | `google/gemma-3-27b-it` |
| LLaVA-1.6 | 34B / Mistral-7B / Vicuna-13B | `llava-hf/llava-v1.6-34b-hf` |
| MiniCPM-V | 4.5 | `openbmb/MiniCPM-V-4_5` |
| InternVL3.5 | 1B / 2B / 4B / 8B / 14B / 38B | `OpenGVLab/InternVL3_5-8B` |
| Qwen2.5-VL | 3B / 7B / 32B / 72B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| BLIP-2 | OPT-2.7B / OPT-6.7B | `Salesforce/blip2-opt-2.7b` |

## Download links

- Fuyu-8B: <https://huggingface.co/adept/fuyu-8b>
- DeepSeek-VL: <https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat>, <https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat>
- Gemma-3-IT: <https://huggingface.co/google/gemma-3-4b-it> (also 12b / 27b)
- LLaVA-1.6: <https://huggingface.co/llava-hf/llava-v1.6-34b-hf> (also mistral-7b / vicuna-13b)
- MiniCPM-V-4.5: <https://huggingface.co/openbmb/MiniCPM-V-4_5>
- InternVL3.5: <https://huggingface.co/OpenGVLab/InternVL3_5-8B> (1B–38B)
- Qwen2.5-VL: <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct> (3B–72B)
- BLIP-2: <https://huggingface.co/Salesforce/blip2-opt-2.7b>, <https://huggingface.co/Salesforce/blip2-opt-6.7b>
- GPT / Gemini: <https://platform.openai.com/docs/models>, <https://ai.google.dev/gemini-api/docs/models>

## How dispatch works

`ModelFactory` parses the id into a family + size and instantiates that family's wrapper
with the requested `mode` / `quant` / `temperature`. Coverage: every family has
`default` + `reasoning`; local families add `temp`; `quant` exists for blip2 / internvl /
llava. To add a model, see [`../multibbq/models/README.md`](../multibbq/models/README.md).
