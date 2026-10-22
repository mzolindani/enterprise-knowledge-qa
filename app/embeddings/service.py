
import math
from collections.abc import Sequence

from app.chunking.models import DocumentChunk
from app.embeddings.models import EmbeddedChunk
from app.embeddings.provider import EmbeddingProvider


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    provider: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    """Embed document chunks while preserving their source metadata."""
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError("Chunks must not contain empty text.")

    vectors = provider.embed_texts(texts)

    if len(vectors) != len(chunks):
        raise ValueError(
            "Embedding count must match the number of input chunks."
        )

    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError(
            "All embeddings must have the same non-zero dimension."
        )

    embedded_chunks: list[EmbeddedChunk] = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                f"Invalid embedding values for chunk {chunk.chunk_id}."
            )

        embedded_chunks.append(
            EmbeddedChunk(
                chunk=chunk,
                embedding=tuple(float(value) for value in vector),
            )
        )

    return embedded_chunks
