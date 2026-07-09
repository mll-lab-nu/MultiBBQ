"""Qwen2.5-VL wrapper (3B / 7B / 32B / 72B Instruct).

Collapses qwen2_5vl_model.py, qwen2_5vl_model_reasoning.py and
qwen2_5vl_model_temp.py into one class. The historical differences were:

    reasoning: sys_prompt = system_msg + think_rule   (direct concat, no space)
    temp:      generate(..., do_sample=True, temperature=t)  [via gen_kwargs]

Note: the original quant factory mapped Qwen to this *unquantized* class, so
``quant=True`` is intentionally a no-op here (behavior preserved).
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseModel


class Qwen2_5VLModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def run(self, image_path: str, user_prompt: str, system_msg: str = None):
        sys_prompt = system_msg + self.think_rule if self.reasoning else system_msg
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user",
             "content": [{"type": "image", "image": image_path},
                         {"type": "text", "text": user_prompt}], }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return output_text
