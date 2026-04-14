import logging
import time

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache
from PIL import Image
from transformers import AutoProcessor, AutoModel

from ragrag.models import MultiVector


logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_quantization(setting: str, device: str) -> str:
    """Resolve a quantization setting to a concrete strategy.

    Returns one of: 'none', '8bit', '4bit'. 'auto' picks based on free VRAM:
    a 4 B vision-language model with high visual-token budget needs roughly
    ~5 GiB free for 4-bit, ~7 GiB for 8-bit, and ~11 GiB for bf16 once
    activations are accounted for.
    """
    s = (setting or "auto").lower()
    if s == "auto":
        if device != "cuda":
            return "none"
        try:
            free_mib = torch.cuda.mem_get_info(0)[0] // 1024**2
        except Exception:
            return "8bit"
        if free_mib >= 12 * 1024:
            return "none"
        if free_mib >= 7 * 1024:
            return "8bit"
        return "4bit"
    if s in {"none", "8bit", "4bit"}:
        return s
    logger.warning("Unknown quantization '%s', falling back to 'none'", setting)
    return "none"


class ColQwenEmbedder:
    def __init__(
        self,
        model_id: str,
        max_visual_tokens: int = 16384,
        quantization: str = "auto",
    ):
        """Load model and processor. Takes ~2-5 min on CPU with swap."""
        # Remember init args so ``reload()`` can rebuild without the
        # caller having to pass them again.
        self.model_id = model_id
        self.max_visual_tokens = int(max_visual_tokens)
        self.quantization = quantization
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        """Load model + processor. Used by __init__ and reload()."""
        model_id = self.model_id
        max_visual_tokens = self.max_visual_tokens
        quantization = self.quantization
        t0 = time.time()
        _cached = try_to_load_from_cache(model_id, "config.json")
        local_only = isinstance(_cached, str)
        logger.info("Model cache check: %s (local_only=%s)", model_id, local_only)
        logger.info("Loading model %s (local_files_only=%s)", model_id, local_only)

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                max_num_visual_tokens=max_visual_tokens,
                local_files_only=local_only,
            )
        except FileNotFoundError:
            logger.warning("Cache miss for processor, retrying without local_files_only")
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                max_num_visual_tokens=max_visual_tokens,
                local_files_only=False,
            )

        device = _detect_device()
        dtype = torch.float16 if device == "mps" else torch.bfloat16
        quant = _resolve_quantization(quantization, device)
        logger.info("Embedder device=%s quantization=%s", device, quant)

        bnb_config = None
        if quant in {"8bit", "4bit"} and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                if quant == "8bit":
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
            except ImportError:
                logger.warning("bitsandbytes not available; disabling quantization")
                quant = "none"

        def _load_model(local_files_only: bool):
            try:
                kwargs = dict(
                    attn_implementation="sdpa",
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                )
                if bnb_config is not None:
                    # bitsandbytes manages dtype internally; passing dtype here
                    # would conflict with the quantization config.
                    kwargs["quantization_config"] = bnb_config
                else:
                    kwargs["dtype"] = dtype

                if device == "mps":
                    model = AutoModel.from_pretrained(model_id, **kwargs).eval().to("mps")
                elif device == "cuda":
                    # device_map="auto" lets accelerate spill layers that don't fit
                    # in VRAM onto CPU/disk instead of OOMing. On a small GPU this is
                    # the difference between partial GPU acceleration and pure CPU.
                    free_vram_mib = torch.cuda.mem_get_info(0)[0] // 1024**2
                    max_mem_mib = max(free_vram_mib - 512, 1024)
                    model = AutoModel.from_pretrained(
                        model_id,
                        device_map="auto",
                        max_memory={0: f"{max_mem_mib}MiB", "cpu": "24GiB"},
                        **kwargs,
                    ).eval()
                else:
                    model = AutoModel.from_pretrained(model_id, device_map=device, **kwargs).eval()
                return model
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU out of memory, falling back to CPU")
                # Release any partial GPU allocation from the failed load before
                # retrying on CPU, otherwise VRAM stays pinned for the session.
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                return AutoModel.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    trust_remote_code=True,
                    device_map="cpu",
                    local_files_only=local_files_only,
                ).eval()

        try:
            self.model = _load_model(local_only)
        except (FileNotFoundError, OSError, RuntimeError) as ex:
            if local_only:
                logger.warning("Local cache incomplete/unusable (%s), retrying with network", ex)
                self.model = _load_model(False)
            else:
                raise

        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def _forward(self, batch: dict) -> list[MultiVector]:
        """Run one forward pass and return per-item MultiVectors (padding removed).

        ColQwen3's text processor left-pads the batch, so the "real" tokens for
        a shorter item live in the tail of the sequence. We select by attention
        mask instead of slicing by length, which handles left- or right-padded
        inputs identically.

        Returns a list of float32 ``np.ndarray`` (shape ``(n_tokens_i, 320)``).
        Numpy is ~8x more memory-efficient than nested Python float lists for
        the multivector representation — important when batching many text
        chunks or holding image embeddings during streaming ingest.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)
        embeddings = out.embeddings
        attn = batch.get("attention_mask")
        results: list[MultiVector] = []
        for i in range(embeddings.shape[0]):
            if attn is not None:
                row_tensor = embeddings[i][attn[i].bool()]
            else:
                row_tensor = embeddings[i]
            arr = row_tensor.to(device="cpu", dtype=torch.float32).numpy()
            # Detach from any underlying tensor so the caller fully owns it.
            results.append(np.ascontiguousarray(arr))
        return results

    def embed_text_chunks(self, texts: list[str]) -> list[MultiVector]:
        """Embed a list of text chunks in a single forward pass."""
        if not texts:
            return []
        t0 = time.time()
        batch = self.processor.process_texts(texts=list(texts))
        vectors = self._forward(batch)
        logger.debug("Text batch: %d items in %.2fs", len(vectors), time.time() - t0)
        return vectors

    def embed_images(self, images: list[Image.Image]) -> list[MultiVector]:
        """Embed a list of images in a single forward pass."""
        if not images:
            return []
        t0 = time.time()
        batch = self.processor.process_images(images=list(images))
        vectors = self._forward(batch)
        logger.debug("Image batch: %d items in %.2fs", len(vectors), time.time() - t0)
        return vectors

    def embed_query_text(self, query: str) -> MultiVector:
        """Embed a search query into multivector."""
        return self.embed_text_chunks([query])[0]

    def embed_text_chunk(self, text: str) -> MultiVector:
        """Embed a single text document chunk into multivector."""
        return self.embed_text_chunks([text])[0]

    def embed_image(self, image: Image.Image) -> MultiVector:
        """Embed a single image into multivector."""
        return self.embed_images([image])[0]

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (320 for ColQwen3)."""
        return 320

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        """Release GPU/CPU memory held by the model. Idempotent.

        The ingest pipeline calls this before loading the VLM topic chunker
        so both 2.5 GB-class models don't compete for VRAM on small cards.
        Call :meth:`reload` to bring the model back.
        """
        if self.model is None:
            return
        try:
            del self.model
        except Exception:
            pass
        self.model = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        logger.info("ColQwen3 embedder unloaded")

    def reload(self) -> None:
        """Re-run the load path using the original init args. Idempotent."""
        if self.model is not None:
            return
        self._load()
