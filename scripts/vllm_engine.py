from __future__ import annotations

import re
import json
import orjson
from typing import List, Optional

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# JSON cleaning helper
# ---------------------------------------------------------------------------

def clean_to_json_string(text: str, task: str = "running"):
    """
    Normalise raw LLM output to a clean JSON string (task='running')
    or a parsed Python object (task='extraction').

    Steps
    -----
    1. Strip <think>...</think> blocks (Qwen3 reasoning traces).
    2. For 'running' tasks: extract the first JSON object/array as a string.
    3. For 'extraction' tasks: parse and return the Python object directly.
    """
    # 1. Remove thinking block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if task != "extraction":
        # Return raw JSON string (caller will parse with valid_json())
        obj = re.search(r"\{.*\}", text, re.DOTALL)
        if obj:
            return obj.group(0).strip()
        arr = re.search(r"\[.*\]", text, re.DOTALL)
        if arr:
            return arr.group(0).strip()
        return text.strip()

    # extraction → return parsed object immediately
    for loader in (orjson.loads, json.loads):
        try:
            return loader(text)
        except Exception:
            pass

    # fallback: extract JSON substring
    for start_c, end_c in (("[", "]"), ("{", "}")):
        try:
            s = text.find(start_c)
            e = text.rfind(end_c) + 1
            if s != -1 and e > 0:
                return orjson.loads(text[s:e])
        except Exception:
            pass

    return {"raw": text}


# ---------------------------------------------------------------------------
# Text client
# ---------------------------------------------------------------------------

class VllmClient:
    """
    In-process vLLM client for text generation using the HuggingFace chat
    template stored in the model's tokenizer config.

    Parameters
    ----------
    model_path             : path to the model weights directory
    temperature            : default sampling temperature
    max_tokens             : default max OUTPUT tokens (not including prompt)
    tensor_parallel_size   : number of GPUs for tensor parallelism
    gpu_memory_utilization : fraction of GPU VRAM to pre-allocate
                             (0.75-0.80 for main model, 0.15 for negation)
    max_batch_size         : internal batch size cap for chat_batch()
    enable_thinking        : pass enable_thinking to Qwen3 chat template
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 5000,
        max_model_len: int = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.78,
        max_batch_size: int = 256,
        enable_thinking: bool = False,
        attention_backend: str = 'eager'
    ):
        self.model_path             = model_path
        self.temperature            = temperature
        self.max_tokens             = max_tokens
        self.tensor_parallel_size   = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_batch_size         = max_batch_size
        self.enable_thinking        = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )
        if max_model_len is not None:
            self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            #attention_backend=attention_backend
        )
        else:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                #attention_backend=attention_backend
            )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, system: str, user: str,
                      enable_thinking: Optional[bool] = None) -> str:
        """Apply the tokenizer's chat template."""
        et = self.enable_thinking if enable_thinking is None else enable_thinking
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # enable_thinking kwarg only applies to Qwen3; other tokenizers ignore it
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=et,
            )
        except TypeError:
            # tokenizer does not accept enable_thinking
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _make_params(self, temperature: Optional[float],
                     max_tokens: Optional[int]) -> SamplingParams:
        """Return a SamplingParams, overriding defaults if requested."""
        t = temperature if temperature is not None else self.temperature
        m = max_tokens  if max_tokens  is not None else self.max_tokens
        if t == self.temperature and m == self.max_tokens:
            return self.sampling_params          # reuse cached object
        return SamplingParams(temperature=t, max_tokens=m)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task: str = "running",
    ):
        """Single-sample inference. Returns cleaned JSON string or parsed object."""
        prompt  = self._build_prompt(system, user)
        params  = self._make_params(temperature, max_tokens)
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        raw     = outputs[0].outputs[0].text.strip()
        return clean_to_json_string(raw, task)

    def chat_batch(
        self,
        system: str,
        users: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        task: str = "running",
    ) -> List:
        """Batched inference. Returns list aligned with `users`."""
        bs      = max_batch_size or self.max_batch_size
        params  = self._make_params(temperature, max_tokens)
        results = []
        for i in range(0, len(users), bs):
            chunk   = users[i:i + bs]
            prompts = [self._build_prompt(system, u) for u in chunk]
            outs    = self.llm.generate(prompts, params, use_tqdm=False)
            for o in outs:
                raw = o.outputs[0].text.strip()
                results.append(clean_to_json_string(raw, task))
        return results


# ---------------------------------------------------------------------------
# Vision client
# ---------------------------------------------------------------------------

class VllmVisionClient:
    """
    In-process vLLM client for multimodal vision-language inference.

    Supports image + text prompts via the HuggingFace chat template.
    Designed for models like LLaMA-3.2-11B-Vision-Instruct and Qwen-VL.

    gpu_memory_utilization should be set to ~0.78 when this is the only
    model loaded, or lower when sharing the GPU with a text client.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.78,
        max_batch_size: int = 4,
    ):
        self.model_path             = model_path
        self.temperature            = temperature
        self.max_tokens             = max_tokens
        self.tensor_parallel_size   = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_batch_size         = max_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=False
        )

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user},
                ],
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    @staticmethod
    def _preprocess_image(img_path: str, target_size: int = 112) -> Image.Image:
        """Resize image to a fixed square canvas (preserves aspect ratio)."""
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((target_size, target_size), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        left = (target_size - img.width) // 2
        top  = (target_size - img.height) // 2
        canvas.paste(img, (left, top))
        return canvas

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, system: str, user: str, image_path: str) -> str:
        """Single image + text inference."""
        prompt = self._build_prompt(system, user)
        image  = self._preprocess_image(image_path)
        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            self.sampling_params,
        )
        return outputs[0].outputs[0].text.strip()

    def chat_batch(
        self,
        system: str,
        users: List[str],
        image_paths: List[str],
    ) -> List[str]:
        """Batched image + text inference."""
        results = []
        for i in range(0, len(users), self.max_batch_size):
            chunk_users  = users[i:i + self.max_batch_size]
            chunk_images = image_paths[i:i + self.max_batch_size]

            inputs = [
                {
                    "prompt": self._build_prompt(system, u),
                    "multi_modal_data": {
                        "image": self._preprocess_image(img_path)
                    },
                }
                for u, img_path in zip(chunk_users, chunk_images)
            ]

            outs = self.llm.generate(inputs, self.sampling_params)
            results.extend(o.outputs[0].text.strip() for o in outs)

        return results
