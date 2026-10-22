
from pathlib import Path

from app.chunking import chunk_document
from app.embeddings import (
    EmbeddingProvider,
    LocalSentenceTransformerProvider,
    embed_chunks,
)
from app.ingestion import ingest_file
from app.retrieval import BM25Retriever, ChromaVectorStore


class DocumentIndexingPipeline:
    """Ingest, chunk, embed, and index a document."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else LocalSentenceTransformerProvider()
        )

    def index_file(self, file_path: str | Path) -> int:
        """Ingest, embed, and index a PDF or DOCX file."""
        document = ingest_file(file_path)
        chunks = chunk_document(document)

        if not chunks:
            self.vector_store.replace_document(
                document.document_id,
                [],
            )

            if self.bm25_retriever is not None:
                self.bm25_retriever.replace_document(
                    document.document_id,
                    [],
                )

            return 0

        embedded_chunks = embed_chunks(
            chunks,
            self.embedding_provider,
        )

        self.vector_store.replace_document(
            document.document_id,
            embedded_chunks,
        )

        if self.bm25_retriever is not None:
            self.bm25_retriever.replace_document(
                document.document_id,
                chunks,
            )

        return len(embedded_chunks)
