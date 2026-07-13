"""Gemini (Vertex AI) wrapper: gemini-2.5-pro / -flash / -flash-lite.

Collapses gemini_model.py and gemini_model_reasoning.py (the live retry-loop
implementations; earlier commented-out drafts dropped). API models have no
'temp' or 'quant' variant, so those are rejected in __init__. The historical
default-vs-reasoning differences were:

    default:   max_output_tokens=20,
               thinking_config=ThinkingConfig(include_thoughts=False,
                   thinking_budget=0 if 'flash' in model name else 128),
               response_schema=Schema(STRING, enum=['A','B','C'])  (enforced)
    reasoning: system_msg = f'{system_msg}\\n{think_rule}',
               max_output_tokens=3000,
               thinking_config=ThinkingConfig(include_thoughts=True,
                                              thinking_budget=-1),
               response_schema DISABLED (free text; enum would clip thoughts)

Both keep response_mime_type='text/x.enum', temperature=0, all safety
categories BLOCK_NONE, and the same 429/backoff retry loop that returns
``None`` after 12 failed tries (the eval loop's retry depends on that).
"""
from google import genai
from google.genai import types
from google.genai import errors
import base64
import os
import time
import random
import logging

from .base import BaseModel


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


_SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
]


class GeminiModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        if quant or mode == "temp":
            raise ValueError("GeminiModel supports only 'default' / 'reasoning' modes")
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        self.model_name = model_id
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "Set GOOGLE_CLOUD_PROJECT to your Vertex AI project id "
                "(see docs/getting-started/installation.md, API credentials)"
            )
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        base64_image = encode_image(image_path)

        if self.reasoning:
            system_msg = f'{system_msg}\n{self.think_rule}'
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=3000,
                safety_settings=_SAFETY_SETTINGS,
                # return one letter only
                response_mime_type='text/x.enum',
                # reasoning historically disabled the A/B/C response_schema
                # so the model can emit its thoughts freely
                thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=-1),
            )
        else:
            # minimal thinking - flash/flash-lite: 0, pro: 128
            if 'flash' in self.model_name:
                thinking_budget = 0
            else:
                thinking_budget = 128
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=20,
                safety_settings=_SAFETY_SETTINGS,
                # return one letter only
                response_mime_type='text/x.enum',
                response_schema=types.Schema(
                    type=types.Type.STRING,
                    enum=['A', 'B', 'C'],
                ),
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=thinking_budget),
            )

        max_tries = 12
        base_sleep = 2.0
        max_sleep = 120.0

        last_err = None
        for t in range(max_tries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=f'{user_prompt}\n{system_msg}'),
                                types.Part.from_bytes(
                                    data=base64.b64decode(base64_image),
                                    mime_type="image/png",
                                )
                            ],
                        )
                    ],
                    config=config,
                )

                return getattr(response, "text", None)

            except errors.ClientError as e:
                last_err = e
                s = str(e)
                if ("429" in s) or ("RESOURCE_EXHAUSTED" in s) or ("Resource exhausted" in s):
                    sleep = min(max_sleep, base_sleep * (2 ** t)) * (0.8 + 0.4 * random.random())
                    logging.info(f"[Gemini 429] sleep {sleep:.1f}s then retry: {e}")
                    time.sleep(sleep)
                    continue
                raise

            except Exception as e:
                last_err = e
                sleep = min(10.0, base_sleep * (0.8 + 0.4 * random.random()))
                logging.info(f"[Gemini ERR] sleep {sleep:.1f}s then retry: {e}")
                time.sleep(sleep)
                continue

        logging.info(f"[Gemini FAIL] exceeded retries, last error: {last_err}")
        return None
