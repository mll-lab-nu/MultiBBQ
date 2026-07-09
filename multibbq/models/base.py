"""Shared base class for all MLLM wrappers.

Replaces the historical trio base_model.py / base_model_reasoning.py /
base_model_temp.py, which differed only in ``gen_kwargs``:

    default:   {'do_sample': False, 'max_new_tokens': 20}
    reasoning: {'do_sample': False, 'max_new_tokens': 2000}
    temp:      {'do_sample': True,  'max_new_tokens': 20, 'temperature': t}

The three axes are orthogonal and are now constructor parameters:

    mode:        'default' | 'reasoning' | 'temp'
                 - 'reasoning' additionally injects ``self.think_rule`` into the
                   prompt/system message (each wrapper reproduces its original
                   injection point exactly).
    quant:       load the HF checkpoint quantized. Per-model precision follows
                 the original *_quant wrappers (blip2/internvl = 8-bit,
                 llava = 4-bit); models without a quant variant ignore it.
    temperature: only used when mode == 'temp' (sampled decoding).

``run()`` has a single signature everywhere: run(image_path, user_prompt, system_msg).
"""


class BaseModel:
    MODES = ("default", "reasoning", "temp")

    def __init__(self, model_path: str, device: str = "cuda",
                 mode: str = "default", quant: bool = False, temperature=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.name = None
        self.size = None
        self.model_path = model_path
        self.device = device
        self.mode = mode
        self.quant = quant
        # exact string used by every historical *_reasoning wrapper
        self.think_rule = "No repetition or extra explanation."
        if mode == "reasoning":
            self.gen_kwargs = {"do_sample": False, "max_new_tokens": 2000}
        elif mode == "temp":
            if temperature is None:
                raise ValueError("mode='temp' requires a temperature")
            self.gen_kwargs = {"do_sample": True, "max_new_tokens": 20,
                               "temperature": float(temperature)}
        else:
            self.gen_kwargs = {"do_sample": False, "max_new_tokens": 20}

    @property
    def reasoning(self) -> bool:
        return self.mode == "reasoning"

    def run(self, image_path: str, user_prompt: str, system_msg: str):
        raise NotImplementedError()
