"""Gemma 3 wrapper (4B / 12B / 27B it).

Collapses gemma_model.py, gemma_model_reasoning.py and gemma_model_temp.py
into one class. The historical differences were:

    reasoning: sys_prompt = think_rule + system_msg   (think_rule FIRST,
               direct concat, no space -- opposite order to qwen_vl)
    temp:      generate(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]

There was never a gemma quant variant, so ``quant=True`` raises.
"""
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from .base import BaseModel


class Gemma3Model(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant:
            raise ValueError("Gemma3Model has no quantized variant (quant=True unsupported)")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        sys_prompt = self.think_rule + system_msg if self.reasoning else system_msg
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.gen_kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
