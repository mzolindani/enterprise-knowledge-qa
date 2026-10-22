
import math  # noqa: F401
from collections.abc import Sequence
from pathlib import Path

import chromadb

from app.embeddings.models import EmbeddedChunk
from app.retrieval.models import RetrievalResult


class ChromaVectorStore:
    """Store embedded document chunks and retrieve them by similarity."""


    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        client=None,
        persist_directory: str | Path | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty.")

        if client is not None and persist_directory is not None:
            raise ValueError(
                "Provide either client or persist_directory, not both."
            )

        if client is not None:
            self.client = client
        elif persist_directory is not None:
            directory = Path(persist_directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(directory)
            )
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )


    def add_chunks(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Store embedded chunks and their source metadata."""
        if not chunks:
            return

        ids, documents, embeddings, metadatas = self._prepare_chunks(chunks)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def _prepare_chunks(
        self,
        chunks: Sequence[EmbeddedChunk],
    ) -> tuple[
        list[str],
        list[str],
        list[list[float]],
        list[dict[str, str | int]],
    ]:
        """Validate chunks and prepare ChromaDB records."""
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str | int]] = []

        for item in chunks:
            if not item.chunk.text.strip():
                raise ValueError(
                    f"Chunk {item.chunk_id} must not contain empty text."
                )

            if not item.embedding:
                raise ValueError(
                    f"Chunk {item.chunk_id} has an empty embedding."
                )

            if not all(
                isinstance(value, (int, float))
                and __import__("math").isfinite(value)
                for value in item.embedding
            ):
                raise ValueError(
                    f"Chunk {item.chunk_id} has invalid embedding values."
                )

            ids.append(item.chunk_id)
            documents.append(item.chunk.text)
            embeddings.append(list(item.embedding))

            metadata: dict[str, str | int] = {
                "document_id": item.document_id,
                "document_name": item.document_name,
                "chunk_index": item.chunk.chunk_index,
            }

            if item.page_number is not None:
                metadata["page_number"] = item.page_number

            if item.section is not None:
                metadata["section"] = item.section

            metadatas.append(metadata)

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) > 1 or 0 in dimensions:
            raise ValueError(
                "All embeddings must have the same non-zero dimension."
            )

        return ids, documents, embeddings, metadatas

    
    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Replace all indexed chunks belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        # Validate ownership before making any changes.
        if any(item.document_id != document_id for item in chunks):
            raise ValueError(
                "All chunks must belong to the specified document."
            )

        # Prepare and validate all replacement records first.
        ids, documents, embeddings, metadatas = self._prepare_chunks(chunks)

        existing = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )
        existing_ids = existing["ids"]

        # An empty replacement means remove the document from the index.
        if not chunks:
            if existing_ids:
                self.collection.delete(ids=existing_ids)
            return

        # Write replacement records before removing obsolete records.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        obsolete_ids = set(existing_ids) - set(ids)

        if obsolete_ids:
            self.collection.delete(ids=list(obsolete_ids))

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the most similar chunks for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        if not all(
            isinstance(value, (int, float))
            and __import__("math").isfinite(value)
            for value in query_embedding
        ):
            raise ValueError(
                "query_embedding must contain finite numeric values."
            )

        if self.collection.count() == 0:
            return []

        response = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            similarity = 1.0 - float(distance)

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata["document_name"]),
                    text=text or "",
                    page_number=metadata.get("page_number"),
                    section=metadata.get("section"),
                    score=similarity,
                )
            )

        return results

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all chunks from the collection."""
        ids = self.collection.get(include=[])["ids"]

        if ids:
            self.collection.delete(ids=ids)
