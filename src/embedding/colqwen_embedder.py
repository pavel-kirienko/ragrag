import logging
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from src.models import MultiVector


logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ColQwenEmbedder:
    def __init__(self, model_id: str, max_visual_tokens: int = 1280):
        """Load model and processor. Takes ~2-5 min on CPU with swap."""
        logger.info(f"Loading ColQwen3 model: {model_id}")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            max_num_visual_tokens=max_visual_tokens,
        )
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU out of memory, falling back to CPU")
            self.model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                device_map="cpu",
            ).eval()
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def _embed_text(self, text: str) -> MultiVector:
        t0 = time.time()
        batch = self.processor.process_texts(texts=[text])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)
        # out.embeddings: (1, seq_len, 320), L2-normalized
        result = out.embeddings[0].to(device="cpu", dtype=torch.float32).tolist()
        logger.debug(f"Text embed: {len(result)} tokens in {time.time() - t0:.2f}s")
        return result

    def embed_query_text(self, query: str) -> MultiVector:
        """Embed a search query into multivector. Uses process_texts()."""
        return self._embed_text(query)

    def embed_text_chunk(self, text: str) -> MultiVector:
        """Embed a text document chunk into multivector. Uses process_texts()."""
        return self._embed_text(text)

    def embed_image(self, image: Image.Image) -> MultiVector:
        """Embed an image into multivector. Uses process_images()."""
        t0 = time.time()
        batch = self.processor.process_images(images=[image])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self.model(**batch)
        result = out.embeddings[0].to(device="cpu", dtype=torch.float32).tolist()
        logger.debug(f"Image embed: {len(result)} tokens in {time.time() - t0:.2f}s")
        return result

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (320 for ColQwen3)."""
        return 320
