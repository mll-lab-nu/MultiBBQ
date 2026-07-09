"""DeepSeek-VL wrapper (deepseek-ai/deepseek-vl-1.3b-chat / -7b-chat).

Collapses deepseekvl_model.py, deepseekvl_model_reasoning.py and
deepseekvl_model_temp.py into one class. The historical differences were:

    reasoning: think_rule prepended (with a trailing space) to the content
               string: f'{think_rule}{system_msg}\\n<image_placeholder>{user_prompt}'
               where think_rule = 'No repetition or extra explanation. '
    temp:      generate(..., max_new_tokens=20, do_sample=True, temperature=t)
               [via gen_kwargs]

Requires the external ``deepseek_vl`` package (vendored in the old repo).
There was no quant variant, so ``quant=True`` raises.
"""
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from .base import BaseModel


class DeepSeekVLModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant:
            raise ValueError("DeepSeek-VL has no quant variant")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        vl_chat_processor = self.vl_chat_processor
        tokenizer = self.tokenizer
        vl_gpt = self.vl_gpt

        if self.reasoning:
            # original: think_rule = 'No repetition or extra explanation. '
            # (trailing space) prepended to the content string
            content = f"{self.think_rule} {system_msg}\n<image_placeholder>{user_prompt}"
        else:
            content = f"{system_msg}\n<image_placeholder>{user_prompt}"

        conversation = [
            {
                "role": "User",
                "content": content,
                "images": [image_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            **self.gen_kwargs,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
