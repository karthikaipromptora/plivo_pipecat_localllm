#
# Upload endpoint — accepts a PDF, chunks + embeds it (GPU), stores in Qdrant.
# Returns a kb_id that you pass in the body when making an outbound call.
#

import io
import os
import uuid
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

load_dotenv(override=True)

router = APIRouter()

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768          # bge-base output dimensions
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # character overlap between chunks


# ── Singletons ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _embedding_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model for upload on {device.upper()}...")
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def _qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL must be set in .env")
    api_key = os.getenv("QDRANT_API_KEY") or None  # None = no auth (local Docker)
    return QdrantClient(url=url, api_key=api_key)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_collection(client: QdrantClient):
    """Create collection + payload index if they don't already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{COLLECTION_NAME}'")

        # Index kb_id for fast per-agent filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="kb_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload index on 'kb_id'")


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(..., description="PDF file to index"),
    name: Optional[str] = Form(None, description="Human-readable name for this KB"),
    kb_id: Optional[str] = Form(None, description="Reuse an existing kb_id to append chunks"),
):
    """Upload a PDF and index it into Qdrant.

    Returns:
        kb_id  — pass this in `body.kb_id` when calling POST /start
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    knowledge_base_id = kb_id or str(uuid.uuid4())
    doc_name = name or file.filename

    logger.info(f"Uploading '{doc_name}' → kb_id={knowledge_base_id}")

    # 1. Read PDF
    pdf_bytes = await file.read()

    # 2. Extract text
    try:
        text = _extract_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {e}")

    if not text:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    # 3. Chunk
    chunks = _chunk_text(text)
    logger.info(f"Split into {len(chunks)} chunks")

    # 4. Embed all chunks in one GPU batch
    model = _embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    logger.info(f"Embedded {len(embeddings)} chunks")

    # 5. Upsert into Qdrant
    client = _qdrant_client()
    _ensure_collection(client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
            payload={
                "text": chunk,
                "kb_id": knowledge_base_id,
                "source": doc_name,
            },
        )
        for emb, chunk in zip(embeddings, chunks)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Indexed {len(points)} points for kb_id={knowledge_base_id}")

    return JSONResponse({
        "kb_id": knowledge_base_id,
        "name": doc_name,
        "chunk_count": len(points),
        "status": "indexed",
    })


@router.delete("/upload/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    """Delete all chunks for a given kb_id from Qdrant."""
    client = _qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
        ),
    )
    logger.info(f"Deleted all chunks for kb_id={kb_id}")
    return JSONResponse({"kb_id": kb_id, "status": "deleted"})
