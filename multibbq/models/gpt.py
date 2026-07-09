"""OpenAI wrappers: GPT-4o and GPT-5 family (gpt-5 / gpt-5-mini / gpt-5-nano).

Collapses gpt4_model.py / gpt4_model_reasoning.py and gpt5_model.py /
gpt5_model_reasoning.py into two classes. API models have no 'temp' or
'quant' variant, so those are rejected in __init__. The historical
default-vs-reasoning differences were:

    GPT4OModel:
        default:   temperature=0, max_output_tokens=20,   system prompt as-is
        reasoning: temperature=0, max_output_tokens=2000,
                   sys_prompt = think_rule + system_msg   (prepend, no space)

    GPT5Model (temperature not supported by the API):
        default:   max_output_tokens=20,   reasoning={"effort": "minimal"}
        reasoning: max_output_tokens=3000, reasoning={"effort": "medium"},
                   system_msg = f'{system_msg}\\n{think_rule}'

Both fall through to ``None`` when no output text can be extracted; the eval
loop's retry depends on that.
"""
import base64
import os

from openai import OpenAI

from .base import BaseModel


def encode_image(image_path):
    # Function to encode the image
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _extract_text(response):
    """Shared Responses-API output handling (identical in all four originals)."""
    text = getattr(response, "output_text", None)
    if text and text.strip():
        return text.strip()

    try:
        data = response.model_dump()
        chunks = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") in ("output_text", "text"):
                        chunks.append(block.get("text", ""))
        merged = "".join(chunks).strip()
        if merged:
            return merged
    except Exception:
        pass
    return None


class GPT4OModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant or mode == "temp":
            raise ValueError("GPT4OModel supports only 'default' / 'reasoning' modes")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.model_name = model_id
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        base64_image = encode_image(image_path)
        # reasoning historically *prepended* the rule with no separator
        sys_prompt = self.think_rule + system_msg if self.reasoning else system_msg
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": sys_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                            "detail": 'low',
                        }
                    ]
                },
            ],
            temperature=0,
            max_output_tokens=2000 if self.reasoning else 20,
        )
        return _extract_text(response)


class GPT5Model(BaseModel):
    # 'gpt-5'
    # 'gpt-5-mini'
    # 'gpt-5-nano'
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant or mode == "temp":
            raise ValueError("GPT5Model supports only 'default' / 'reasoning' modes")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.model_name = model_id
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        base64_image = encode_image(image_path)
        if self.reasoning:
            system_msg = f'{system_msg}\n{self.think_rule}'
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_msg}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                            "detail": 'low',
                        }
                    ]
                },
            ],
            # temperature=0, # this model cannot support the param: temperature
            max_output_tokens=3000 if self.reasoning else 20,
            reasoning={"effort": "medium" if self.reasoning else "minimal"},
        )
        return _extract_text(response)
