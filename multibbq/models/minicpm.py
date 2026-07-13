"""MiniCPM-V wrapper (openbmb/MiniCPM-V-4_5).

Collapses minicpmv_model.py, minicpmv_model_reasoning.py and
minicpmv_model_temp.py into one class. The historical differences were:

    reasoning: enable_thinking=True (False otherwise), think_rule appended as
               a third line: f'{user_prompt}\\n{system_msg}\\n{think_rule}',
               and generation hardcoded do_sample=False, max_new_tokens=3000
               (NOT the shared reasoning default of 2000 - overridden below).
    temp:      chat(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]

The original reasoning wrapper contained a commented-out <think> strip
(re.sub) which was never active; it is preserved as a comment only.
There was no quant variant, so ``quant=True`` raises.
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel

torch.manual_seed(100)


class MiniCPMVModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant:
            raise ValueError("MiniCPM-V has no quant variant")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        if mode == "reasoning":
            # original hardcoded do_sample=False, max_new_tokens=3000
            self.gen_kwargs = {"do_sample": False, "max_new_tokens": 3000}
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        image = Image.open(image_path).convert('RGB')

        enable_thinking = self.reasoning # If `enable_thinking=True`, the thinking mode is enabled.
        stream = False # If `stream=True`, the answer is string

        # First round chat
        if self.reasoning:
            question = f'{user_prompt}\n{system_msg}\n{self.think_rule}'
        else:
            question = f'{user_prompt}\n{system_msg}'
        msgs = [{'role': 'user', 'content': [image, question]}]

        answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            enable_thinking=enable_thinking,
            stream=stream,
            **self.gen_kwargs
        )

        # answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        return answer
