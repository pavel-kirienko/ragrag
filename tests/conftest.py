from __future__ import annotations

import sys
import types
from typing import Any, cast


class _TestEmbedder:
    embedding_dim: int = 4

    def __init__(self, model_id: str, max_visual_tokens: int = 16384, quantization: str = "auto") -> None:
        _ = model_id
        _ = max_visual_tokens
        _ = quantization

    def embed_text_chunk(self, text: str):
        _ = text
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_text_chunks(self, texts):
        return [self.embed_text_chunk(t) for t in texts]

    def embed_query_text(self, query: str):
        _ = query
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_image(self, image: object):
        _ = image
        return [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]

    def embed_images(self, images):
        return [self.embed_image(img) for img in images]


def _detect_device() -> str:
    return "cpu"


if "src.embedding.colqwen_embedder" not in sys.modules:
    module = types.ModuleType("src.embedding.colqwen_embedder")
    module_any = cast(Any, module)
    module_any.ColQwenEmbedder = _TestEmbedder
    module_any._detect_device = _detect_device
    sys.modules["src.embedding.colqwen_embedder"] = module

    import src.embedding as embedding_package

    embedding_package_any = cast(Any, embedding_package)
    embedding_package_any.colqwen_embedder = module
