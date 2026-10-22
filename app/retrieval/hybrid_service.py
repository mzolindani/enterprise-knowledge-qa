
from app.embeddings import EmbeddingProvider
from app.retrieval.bm25_store import BM25Retriever
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_store import ChromaVectorStore


class HybridRetrievalService:
    """Combine dense and lexical retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        embedding_provider: EmbeddingProvider,
        candidate_k: int = 10,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_provider = embedding_provider
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve candidates from both stores and fuse their rankings."""
        if not query.strip():
            raise ValueError("query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embeddings = self.embedding_provider.embed_texts([query])

        if len(query_embeddings) != 1:
            raise ValueError(
                "Embedding provider must return exactly one query embedding."
            )

        query_embedding = query_embeddings[0]

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        vector_results = self.vector_store.search(
            query_embedding,
            top_k=self.candidate_k,
        )

        lexical_results = self.bm25_retriever.search(
            query,
            top_k=self.candidate_k,
        )

        return reciprocal_rank_fusion(
            [vector_results, lexical_results],
            top_k=top_k,
        )
