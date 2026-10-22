
import math
from collections.abc import Sequence
from typing import Any, Protocol

from sentence_transformers import SentenceTransformer


class EmbeddingProvider(Protocol):
    """Interface implemented by embedding providers."""

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class LocalSentenceTransformerProvider:
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "cpu",
        encoder: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.encoder = (
            encoder
            if encoder is not None
            else SentenceTransformer(model_name, device=device)
        )

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed input texts in batches, preserving input order."""
        if not texts:
            return []

        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Input text at index {index} must not be empty."
                )

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])

            encoded = self.encoder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vectors = encoded.tolist()

            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding model returned an unexpected number "
                    "of vectors."
                )

            for vector in vectors:
                if not vector:
                    raise ValueError(
                        "Embedding model returned an empty vector."
                    )

                if not all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError(
                        "Embedding model returned invalid vector values."
                    )

                float_vector = [float(value) for value in vector]
                magnitude = math.sqrt(sum(value**2 for value in float_vector))

                if magnitude == 0:
                    raise ValueError(
                        "Embedding model returned a zero-magnitude vector."
                    )

                normalised_vector = [
                    value / magnitude for value in float_vector
                ]

                embeddings.append(normalised_vector)



        if len(embeddings) != len(texts):
            raise ValueError(
                "Embedding count does not match input count."
            )

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) != 1:
            raise ValueError(
                "Embedding model returned inconsistent vector dimensions."
            )

        return embeddings
