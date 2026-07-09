"""LLaVA-NeXT wrapper (v1.6 vicuna-13b / 34B).

Collapses llava_model.py, llava_model_reasoning.py, llava_model_temp.py and
llava_model_quant.py into one class. The historical differences were:

    reasoning: prompt = f"USER: <image>\\n{user_prompt}\\n\\n{think_rule}\\n\\n{system_msg} ASSISTANT:"
               (think_rule injected between user_prompt and system_msg)
    temp:      generate(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]
    quant:     from_pretrained(..., load_in_4bit=True)  (4-bit, unlike the
               8-bit blip2/internvl quant variants)
"""
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from .base import BaseModel


class LlavaModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        processor = LlavaNextProcessor.from_pretrained(model_id)
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        load_kwargs = {}
        if quant:
            load_kwargs["load_in_4bit"] = True
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            **load_kwargs,
        )
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.eval()
        self.processor = processor
        self.model = model

    def run(self, image_path: str, user_prompt: str, system_msg: str = None):
        if self.reasoning:
            prompt = f"USER: <image>\n{user_prompt}\n\n{self.think_rule}\n\n{system_msg} ASSISTANT:"
        else:
            prompt = f"USER: <image>\n{user_prompt}\n\n{system_msg} ASSISTANT:"
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image,
                                return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        outputs: str = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        outputs = outputs[len(prompt)-len('<image>'):].strip()
        return outputs
