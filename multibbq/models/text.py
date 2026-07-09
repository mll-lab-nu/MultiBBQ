"""Text-only LLM wrapper.

MultiBBQ is built on BBQ, so every record carries the full *unmasked* text
(context, question, options with the real demographic terms). That makes the
benchmark a superset that can also evaluate **language-only LLMs** — the metrics
(`multibbq.metrics`) are modality-agnostic. This wrapper runs a HuggingFace
causal LM on the text alone (no image), which is how the `llm` experiment
measures backbone-LLM fairness.

Generic across HF chat LLMs (Qwen, Llama, Gemma, Mistral, DeepSeek, …) via
`AutoModelForCausalLM` + the tokenizer's chat template. The three BaseModel axes
apply: `mode='reasoning'` appends the think-rule and raises the token budget;
`mode='temp'` samples at `temperature`; `quant` loads in 8-bit if requested.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel


class HFTextModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if quant:
            load_kwargs["load_in_8bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def run(self, image_path, user_prompt, system_msg=None):
        # image_path is accepted for a uniform call signature but ignored (text-only).
        prompt = f"{user_prompt}\n\n{self.think_rule}" if self.reasoning else user_prompt
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, **self.gen_kwargs)
        output = self.tokenizer.decode(
            generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return output
