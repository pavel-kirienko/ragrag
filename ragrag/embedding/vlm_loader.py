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


def plan_vlm_placement(
    *,
    embedder_resident_mib: int = 3000,
    vlm_alone_threshold_mib: int = 3000,
    vlm_coexist_threshold_mib: int = 6000,
) -> str:
    """Decide whether the VLM should run on GPU (coexist / swap) or CPU.

    Returns one of:
      * ``"cuda_coexist"`` — enough free VRAM for the VLM alongside the
        embedder; no swap needed.
      * ``"cuda_swap"``    — not enough for both, but enough for the VLM
        alone after unloading the embedder.
      * ``"cpu"``          — not enough VRAM for the VLM even alone; the
        caller should load the VLM on CPU and leave the embedder alone.

    Thresholds are intentionally generous: Qwen2.5-VL-3B weights are
    ~2.5 GB at 4-bit, cuBLAS workspace is another ~512 MB, and the vision
    encoder needs ~1–2 GB of activations for a multi-image window. An
    8 GB card with a busy X11 desktop holding ~1.5 GB typically ends up
    with ~4–4.5 GB free after the embedder unload, which is right on the
    edge — so we err on the side of CPU to keep the indexing pass from
    crashing mid-run.
    """
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free_mib = torch.cuda.mem_get_info(0)[0] // (1024 * 1024)
    except Exception:
        return "cpu"
    if free_mib >= vlm_coexist_threshold_mib:
        return "cuda_coexist"
    hypothetical_free_after_unload = free_mib + embedder_resident_mib
    if hypothetical_free_after_unload >= vlm_alone_threshold_mib:
        return "cuda_swap"
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
    min_free_vram_mib: int = 3200,
) -> VLMHandle:
    """Load a vision-language model. Returns a ready-to-use ``VLMHandle``.

    If CUDA is available but free VRAM is below ``min_free_vram_mib``, we
    fall back to CPU (bf16) automatically so co-resident models like
    ColQwen3 don't get evicted. Explicit ``device="cuda"`` bypasses this
    safety check.
    """
    import os

    # Prevent HF from making network HEAD requests on every load. The
    # VLM weights are already in the local cache; blocking network I/O
    # keeps startup snappy on flaky networks and avoids 60 s of 10-s
    # timeouts stacking up through urllib3's retry policy.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoProcessor

    explicit_device = device is not None
    chosen_device = device or detect_device()

    # Safety: if CUDA is selected but free VRAM is too tight to fit the VLM
    # AND accompanying model activations, fall back to CPU instead of
    # erroring out the whole indexing pass. Caller can force CUDA with
    # ``device="cuda"`` explicitly.
    if chosen_device == "cuda" and not explicit_device:
        try:
            free_mib = torch.cuda.mem_get_info(0)[0] // (1024 * 1024)
        except Exception:
            free_mib = -1
        if 0 < free_mib < min_free_vram_mib:
            logger.warning(
                "Only %d MiB free VRAM; loading VLM on CPU to preserve GPU room "
                "(threshold: %d MiB). Indexing will be slower.",
                free_mib, min_free_vram_mib,
            )
            chosen_device = "cpu"

    quant = resolve_quantization(quantization, chosen_device)

    logger.info("Loading VLM %s (device=%s, quant=%s)", model_id, chosen_device, quant)
    t0 = time.time()

    cached_config = try_to_load_from_cache(model_id, "config.json")
    prefer_local = isinstance(cached_config, str)
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=prefer_local,
        )
    except (OSError, FileNotFoundError):
        logger.warning("VLM processor cache incomplete; retrying with network fetch")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False,
        )

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
    # transformers < 4.56, then to AutoModelForCausalLM (moondream2
    # ships with auto_map = AutoModelForCausalLM), and finally to
    # plain AutoModel for any remaining custom-code model.
    factory_chain: list[Any] = []
    for factory_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ):
        try:
            factory_chain.append(getattr(__import__("transformers", fromlist=[factory_name]), factory_name))
        except (ImportError, AttributeError):
            continue

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        # Force the eager attention implementation so we don't depend on a
        # cuDNN / flash-attn kernel being registered for the bnb 4-bit
        # compute-dtype combo. The older RTX 30 series + recent cuDNN builds
        # throw "was unable to find an engine to execute this computation"
        # when SDPA tries to dispatch the vision encoder at bf16, and the
        # eager path avoids the whole fused-kernel lookup.
        "attn_implementation": "eager",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        # Pin every module to cuda:0 instead of ``device_map="auto"``.
        # Accelerate's auto heuristic underestimates free VRAM on 8 GB
        # cards (same pattern as the ColQwen3 loader) and starts
        # spilling layers to the CPU or disk, which then blows up
        # under bnb because 4-bit modules can't be dispatched off
        # device. A fixed cuda:0 mapping keeps the whole model on GPU
        # or fails cleanly.
        kwargs["device_map"] = {"": "cuda:0"} if chosen_device == "cuda" else None
    elif chosen_device == "cuda":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": "cuda:0"}
    else:
        kwargs["dtype"] = torch.bfloat16  # CPU bf16 is fine on modern x86

    model = None
    last_err: Exception | None = None
    for factory in factory_chain:
        try:
            model = factory.from_pretrained(
                model_id, local_files_only=prefer_local, **kwargs
            ).eval()
            break
        except (OSError, FileNotFoundError) as exc:
            # Cache miss — retry once with network access before giving
            # up on this factory.
            logger.warning("VLM weights cache incomplete; retrying with network fetch")
            try:
                model = factory.from_pretrained(
                    model_id, local_files_only=False, **kwargs
                ).eval()
                break
            except Exception as retry_exc:  # noqa: BLE001
                last_err = retry_exc
                continue
        except ValueError as exc:
            # The config class isn't registered with this factory
            # (moondream2's HfConfig, etc.). Try the next factory in
            # the chain; the final AutoModel fallback always works
            # because it uses trust_remote_code resolution directly.
            logger.info(
                "VLM %s not recognised by %s, trying next factory (%s)",
                model_id, factory.__name__, exc,
            )
            last_err = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning(
                "VLM load via %s failed: %s — trying next factory",
                factory.__name__, exc,
            )
            continue

    if model is None:
        raise RuntimeError(
            f"Could not load VLM {model_id!r} via any transformers Auto* factory "
            f"(last error: {last_err})"
        )

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
