"""Shared VLM loader for the topic chunker / segmenter / reranker.

The chunker (Phase B) and the reranker (Phase D) both need a small local
vision-language model. We load it once per daemon process and share the
handle. Default model is ``Qwen/Qwen2.5-VL-3B-Instruct`` quantized to 4-bit
via bitsandbytes; on CPU-only hosts we fall back to bf16.

This module is deliberately import-light so that ``ragrag daemon --idle``
keeps booting in under a second. Heavy imports (transformers, torch,
bitsandbytes, PIL) only happen inside ``VLMHandle.load()``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public type aliases
# --------------------------------------------------------------------------- #

PILImageLike = Any  # avoid importing PIL at module load


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

@dataclass
class VLMHandle:
    """Wraps a loaded VLM model + processor so the chunker / reranker share state.

    The handle is constructed via :func:`load_vlm` or directly with already-
    loaded ``model`` / ``processor`` (used by tests with stubs). Call
    :meth:`generate` to run text generation against text + image inputs;
    call :meth:`unload` to release VRAM (idempotent).
    """

    model_id: str
    device: str            # "cuda" / "cpu" / "mps"
    quantization: str      # "4bit" / "8bit" / "none"
    model: Any = None
    processor: Any = None
    loaded_at: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def generate(
        self,
        text: str,
        images: list[PILImageLike] | None = None,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run a single generation call. Returns the assistant's reply text only."""
        if not self.is_loaded:
            raise RuntimeError("VLMHandle is not loaded; call load_vlm first")
        import torch

        # Build the chat-template input.
        content: list[dict[str, Any]] = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt_text],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        )

        # Move tensor inputs to the model's first device. With device_map="auto"
        # accelerate handles cross-device dispatch via hooks.
        first_param = next(self.model.parameters())
        for key, val in list(inputs.items()):
            if hasattr(val, "to"):
                inputs[key] = val.to(first_param.device)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0.0,
                temperature=max(temperature, 0.0001),
            )
        # Strip the prompt prefix.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def unload(self) -> None:
        """Drop model + processor refs and free CUDA cache. Idempotent."""
        if self.model is None and self.processor is None:
            return
        self.model = None
        self.processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
        logger.info("VLM unloaded (%s)", self.model_id)


def detect_device() -> str:
    """Return the best available device name for new VLM loads."""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_quantization(setting: str, device: str) -> str:
    """Pick a quantization strategy for the VLM.

    On CUDA we 4-bit by default to leave room for ColQwen3. On CPU there's
    no useful quantization (bnb requires CUDA), so we run bf16.
    """
    s = (setting or "auto").lower()
    if s == "auto":
        return "4bit" if device == "cuda" else "none"
    if s in {"4bit", "8bit", "none"}:
        return s
    logger.warning("Unknown VLM quantization '%s', falling back to 'none'", setting)
    return "none"


def load_vlm(
    model_id: str,
    *,
    quantization: str = "auto",
    device: str | None = None,
) -> VLMHandle:
    """Load a vision-language model. Returns a ready-to-use ``VLMHandle``."""
    import torch
    from transformers import AutoProcessor

    chosen_device = device or detect_device()
    quant = resolve_quantization(quantization, chosen_device)

    logger.info("Loading VLM %s (device=%s, quant=%s)", model_id, chosen_device, quant)
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    quantization_config = None
    if quant == "4bit" and chosen_device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            logger.warning("bitsandbytes missing, loading VLM without quantization")
            quant = "none"
    elif quant == "8bit" and chosen_device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            quant = "none"

    # Prefer the modern class name; fall back to the older one for
    # transformers < 4.56, and finally to plain AutoModel for safety.
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _AutoModel  # type: ignore[no-redef]
        except ImportError:
            from transformers import AutoModel as _AutoModel  # type: ignore[no-redef]

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "auto"
    elif chosen_device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        kwargs["dtype"] = torch.bfloat16  # CPU bf16 is fine on modern x86

    model = _AutoModel.from_pretrained(model_id, **kwargs).eval()

    handle = VLMHandle(
        model_id=model_id,
        device=chosen_device,
        quantization=quant,
        model=model,
        processor=processor,
        loaded_at=time.time(),
    )
    logger.info("VLM ready in %.1fs (device=%s, quant=%s)", time.time() - t0, chosen_device, quant)
    return handle
