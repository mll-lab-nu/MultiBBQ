"""Fuyu wrapper (adept/fuyu-8b).

Collapses fuyu_model.py, fuyu_model_reasoning.py and fuyu_model_temp.py into
one class. The historical differences were:

    reasoning: prompt = f'{user_prompt}\\n\\n{think_rule}\\n\\n{system_msg} ANSWER:'
               (default/temp used f'{user_prompt}\\n{system_msg} ANSWER:')
    temp:      generate(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]

Fuyu never had a quant wrapper, so ``quant=True`` raises (matching the original
quant factory, which had no fuyu entry).
"""
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import torch

from .base import BaseModel


class FuyuModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant:
            raise ValueError("fuyu has no quantized variant")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.processor = FuyuProcessor.from_pretrained(model_id)
        self.model = FuyuForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",)
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.model.eval()

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        if self.reasoning:
            prompt = f'{user_prompt}\n\n{self.think_rule}\n\n{system_msg} ANSWER:'
        else:
            prompt = f'{user_prompt}\n{system_msg} ANSWER:'
        # prepare inputs for the model
        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image,
                        return_tensors="pt").to(self.device)

        gen_len = self.gen_kwargs['max_new_tokens']
        # autoregressively generate text
        generation_output = self.model.generate(
            **inputs, **self.gen_kwargs)
        generation_text = self.processor.batch_decode(
            generation_output[:, -gen_len:], skip_special_tokens=True)[0].strip()

        idx = 0
        if '\x04' in generation_text:
            idx = generation_text.index('\x04')+1
        return generation_text[idx:]
