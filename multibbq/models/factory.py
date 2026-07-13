"""Unified model factory.

One registry, one entry point. The historical four factories (base / quant /
reasoning / temp) selected between per-variant wrapper *classes*; the variants
are now constructor parameters of a single class per model family, so the
factory just resolves the family and passes ``mode`` / ``quant`` /
``temperature`` through.

Wrappers are imported lazily so that using one model does not require the
dependencies of all the others (e.g. the DeepSeek-VL vendor package or the
OpenAI / Google API clients).
"""
from importlib import import_module

# family key -> (module, class). Keys are the tokens produced by _parse().
_REGISTRY = {
    "llava":        ("multibbq.models.llava", "LlavaModel"),
    "blip2":        ("multibbq.models.blip2", "Blip2Model"),
    "xgen":         ("multibbq.models.blip2", "Blip2Model"),   # historical alias
    "fuyu":         ("multibbq.models.fuyu", "FuyuModel"),
    "InternVL3_5":  ("multibbq.models.internvl", "InternVL3_5Model"),
    "deepseek":     ("multibbq.models.deepseek_vl", "DeepSeekVLModel"),
    "MiniCPM":      ("multibbq.models.minicpm", "MiniCPMVModel"),
    "gemma":        ("multibbq.models.gemma", "Gemma3Model"),
    "Qwen2.5":      ("multibbq.models.qwen_vl", "Qwen2_5VLModel"),
    "Qwen2.5_AWQ":  ("multibbq.models.qwen_vl", "Qwen2_5VLModel"),
    "gemini":       ("multibbq.models.gemini", "GeminiModel"),
}

# families the historical quant factory supported (Qwen mapped to the
# unquantized class - the wrapper keeps that no-op behavior)
_QUANT_FAMILIES = {"llava", "blip2", "InternVL3_5", "Qwen2.5", "Qwen2.5_AWQ"}


def _parse(model_id: str):
    """Derive (family, size) from a HuggingFace/API model id.

    Reproduces the historical parsing exactly, including the AWQ rename.
    """
    parts = model_id.split("/")[-1].split("-")
    name, size = parts[0], parts[-1]
    if size in ("hf", "Instruct", "chat", "it"):
        size = parts[-2]
    if size == "AWQ":
        size = parts[2]
        name = name + "_AWQ"
    return name, size, parts


class ModelFactory:
    """Instantiate the correct MLLM wrapper for a model id.

    Args:
        model_id:    e.g. "OpenGVLab/InternVL3_5-8B", "gpt-5-mini".
        mode:        'default' | 'reasoning' | 'temp'.
        quant:       load the model quantized (families without a quantized
                     variant raise ValueError).
        temperature: required for mode='temp'.
    """

    def create_model(self, model_id: str, mode: str = "default",
                     quant: bool = False, temperature=None, text_only: bool = False):
        name, size, parts = _parse(model_id)

        # Text-only LLM evaluation: any HF causal LM, no vision wrapper.
        if text_only:
            from .text import HFTextModel
            model = HFTextModel(model_id, mode=mode, quant=quant, temperature=temperature)
            if model.name is None:
                model.name = name
            if model.size is None:
                model.size = size
            return model

        if name == "gpt":
            # OpenAI ids: gpt-4o / gpt-5 / gpt-5-mini / gpt-5-nano
            mod = import_module("multibbq.models.gpt")
            if parts[1] == "4o":
                model = mod.GPT4OModel("gpt-4o", mode=mode, quant=quant,
                                       temperature=temperature)
            elif parts[1] == "5":
                api_id = {"mini": "gpt-5-mini", "nano": "gpt-5-nano"}.get(size, "gpt-5")
                model = mod.GPT5Model(api_id, mode=mode, quant=quant,
                                      temperature=temperature)
            else:
                raise ValueError(f"Unsupported gpt model id: {model_id}")
        else:
            spec = _REGISTRY.get(name)
            if spec is None:
                raise ValueError(f"Unsupported model: {name!r} (from {model_id})")
            if quant and name not in _QUANT_FAMILIES:
                raise ValueError(f"Model family {name!r} has no quantized variant")
            module, cls_name = spec
            cls = getattr(import_module(module), cls_name)
            model = cls(model_id, mode=mode, quant=quant, temperature=temperature)

        if model.name is None:
            model.name = name
        if model.size is None:
            model.size = size
        return model
