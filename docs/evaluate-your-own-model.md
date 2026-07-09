# Evaluate your own model

The most common external use case: run MultiBBQ on a model that ships neither in
[models.md](models.md) nor in the registry. You write one wrapper class, register
its family, and the rest of the pipeline (prompting, the retry loop, scoring) is
unchanged. Everything routes through `ModelFactory.create_model` in
[../multibbq/models/factory.py](../multibbq/models/factory.py).

## The wrapper contract

Subclass `BaseModel` ([../multibbq/models/base.py](../multibbq/models/base.py)) and
implement one method:

```python
def run(self, image_path: str, user_prompt: str, system_msg: str):
    ...  # -> the model's raw text answer, or None on API/generation failure
```

- **Return the raw text answer.** The metrics layer parses A/B/C (or an
  unknown-synonym) out of that string, so you do not post-process; see
  `parse_pred` in [../multibbq/metrics/parsers.py](../multibbq/metrics/parsers.py).
- **Return `None` on failure; do not raise.** The reasoning experiments wrap
  `run` in a 3-attempt retry loop that keys on `None` (`_call_model` in
  [../multibbq/inference.py](../multibbq/inference.py) breaks as soon as
  `res is not None`). An exception aborts the whole run; a `None` retries. The
  Gemini wrapper is the reference for "return `None` after retries are
  exhausted".
- **`image_path` is a filesystem path** to the (already resolved) image. Text-only
  models accept it for a uniform signature and ignore it (see
  [../multibbq/models/text.py](../multibbq/models/text.py)).

### The three axes

`BaseModel.__init__(model_path, device="cuda", mode="default", quant=False, temperature=None)`
sets up three orthogonal knobs. Honor them as follows:

| Axis | Values | What your wrapper must do |
|---|---|---|
| `mode` | `default` \| `reasoning` \| `temp` | For `reasoning`, inject `self.think_rule` into the system/user prompt (the base sets `gen_kwargs` to a 2000-token budget). For `temp`, pass `self.gen_kwargs` (which carries `do_sample=True, temperature=…`) to `generate`. |
| `quant` | `bool` | Load the checkpoint quantized if you have a quantized variant; otherwise ignore it (the factory only *reaches* your wrapper with `quant=True` if you opt into `_QUANT_FAMILIES`). |
| `temperature` | `float`\|`None` | Only meaningful when `mode == "temp"`; the base already folds it into `self.gen_kwargs`, so you rarely touch it directly. |

`self.gen_kwargs` is precomputed by the base per mode; pass it straight into your
model's `generate`. Convenience: `self.reasoning` is `True` iff `mode == "reasoning"`.

## Register the family

Add one row to `_REGISTRY` in
[../multibbq/models/factory.py](../multibbq/models/factory.py):

```python
_REGISTRY = {
    ...
    "MyModel": ("multibbq.models.my_model", "MyModel"),   # family key -> (module, class)
}
```

The **family key** must equal the `name` that `_parse` derives from the id.
`_parse(model_id)` splits the last path segment on `-`:

```python
name, size = parts[0], parts[-1]              # "MyModel", "7B" for "org/MyModel-7B"
if size in ("hf", "Instruct", "chat", "it"):  # drops a trailing tag
    size = parts[-2]
```

So `org/MyModel-7B-Instruct` parses to family `MyModel`, size `7B`. Pick your id's
leading token as the key. If your family also has a quantized variant, add the key
to `_QUANT_FAMILIES` in the same file, or `create_model` raises
`"Model family … has no quantized variant"` when `quant=True`.

## Minimal skeleton

`multibbq/models/my_model.py`:

```python
from .base import BaseModel


class MyModel(BaseModel):
    def __init__(self, model_id, mode="default", quant=False, temperature=None):
        super().__init__(model_id, mode=mode, quant=quant, temperature=temperature)
        # load weights / build an API client here (imported lazily by the factory)
        self.model = ...
        self.processor = ...

    def run(self, image_path: str, user_prompt: str, system_msg: str = None):
        sys_prompt = system_msg + self.think_rule if self.reasoning else system_msg
        try:
            # build the request from image_path + user_prompt + sys_prompt,
            # generate with **self.gen_kwargs, decode to text
            return output_text.strip()
        except Exception:
            return None   # let the reasoning retry loop try again
```

[../multibbq/models/qwen_vl.py](../multibbq/models/qwen_vl.py) is the fullest local
exemplar (HF `generate` + chat template + `self.gen_kwargs`);
[../multibbq/models/gpt.py](../multibbq/models/gpt.py) shows the API pattern.

### API keys

API wrappers read credentials from the environment in `__init__`, e.g.
`OpenAI(api_key=os.environ["OPENAI_API_KEY"])` in gpt.py, and
`GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` in gemini.py. Do the same:
read your key with `os.environ[...]` at construction, and keep it out of the id.

### Local vLLM / text-only models

A language-only model subclasses `BaseModel` the same way: accept `image_path`
and ignore it. See [../multibbq/models/text.py](../multibbq/models/text.py)
(`HFTextModel`); it is dispatched via the factory's `text_only=True` path (the
`llm` experiment sets that), not through `_REGISTRY`. A vLLM backend fits the same
contract: build the client in `__init__`, generate in `run`, return text or `None`.

## Validate

Smoke-test on a handful of items before a full sweep:

```bash
# 1. run a couple items of the baseline experiment
multibbq run org/MyModel-7B --experiment main --data_id gpt_image_gen

# 2. score the produced results file (writes *_w_metrics.json)
multibbq score -i results/gpt_image_gen_main/org/MyModel-7B -o /tmp/mm_scored
```

Confirm: the run wrote `results/gpt_image_gen_main/org/MyModel-7B/…json` with non-empty
`pred` strings, and `score` prints an `{"overall": {fairness_score, bias_score,
unk_rate}, "by_category": {…}}` block without parse failures. See
[running.md](running.md) for the full flag set and [metrics.md](metrics.md) for the
scoring pipeline.
