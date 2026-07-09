"""BLIP-2 wrapper (Salesforce/blip2-opt-2.7b / blip2-opt-6.7b).

Collapses blip2_model.py, blip2_model_reasoning.py, blip2_model_temp.py and
blip2_model_quant.py into one class. The historical differences were:

    reasoning: prompt = f'{user_prompt}\\n\\n{think_rule}\\n\\n{system_msg} ANSWER:'
               (default/temp/quant used f"{system_msg} Question: {user_prompt} Answer:")
    temp:      generate(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]
    quant:     from_pretrained(..., load_in_8bit=True) instead of full precision
"""
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from .base import BaseModel


class Blip2Model(BaseModel):
    def __init__(self, model_path, mode="default", quant=False, temperature=None):
        super().__init__(model_path, mode=mode, quant=quant, temperature=temperature)
        self.processor = Blip2Processor.from_pretrained(model_path)
        if quant:
            # 8bit
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map={"": 0},
            )
        else:
            # full precision
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                device_map={"": 0},
            )
        self.model.eval()

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        image = Image.open(image_path)
        if self.reasoning:
            prompt = f'{user_prompt}\n\n{self.think_rule}\n\n{system_msg} ANSWER:'
        else:
            prompt = f"{system_msg} Question: {user_prompt} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            device="cuda", dtype=torch.bfloat16)
        input_token_len = inputs.input_ids.shape[1]
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
        return generated_text
