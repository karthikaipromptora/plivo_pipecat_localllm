#
# Qdrant RAG search — used by the bot during live calls.
# Single collection, filtered by kb_id for per-agent isolation.
# Uses sentence-transformers + PyTorch CUDA for GPU-accelerated embedding.
#

import asyncio
import os
from functools import lru_cache

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768 dims


@lru_cache(maxsize=1)
def _embedding_model():
    """Load SentenceTransformer on GPU once at startup. Reused across all calls."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model on {device.upper()}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info("Embedding model ready.")
    return model


@lru_cache(maxsize=1)
def _async_qdrant_client() -> AsyncQdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL must be set in .env")
    api_key = os.getenv("QDRANT_API_KEY") or None  # None = no auth (local Docker)
    return AsyncQdrantClient(url=url, api_key=api_key)


async def rag_search(query: str, kb_id: str, top_k: int = 5) -> str:
    """Search the knowledge base filtered by kb_id.

    Typical latency (GPU):
      - Embedding:     ~5-15ms  (PyTorch CUDA)
      - Qdrant local:  ~2-5ms
      - Total:         ~10-20ms

    Args:
        query:  Natural language query from the LLM.
        kb_id:  Knowledge base ID passed at call start via body.
        top_k:  Number of top chunks to retrieve.

    Returns:
        Retrieved chunks joined as a string, or a fallback message.
    """
    model = _embedding_model()
    client = _async_qdrant_client()

    # sentence-transformers encode is synchronous — run in thread pool
    loop = asyncio.get_running_loop()
    embedding: list[float] = await loop.run_in_executor(
        None,
        lambda: model.encode(query, normalize_embeddings=True).tolist(),
    )

    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=Filter(
            must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
        ),
        limit=top_k,
    )

    if not response.points:
        logger.debug(f"RAG: no results for kb_id={kb_id!r}, query={query!r}")
        return "No relevant information found."

    chunks = [r.payload["text"] for r in response.points]
    logger.debug(f"RAG: retrieved {len(chunks)} chunks for kb_id={kb_id!r}")
    return "\n\n".join(chunks)
