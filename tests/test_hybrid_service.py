
import pytest

from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.models import RetrievalResult


def make_result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document-001",
        document_name="safety-policy.pdf",
        text=f"Text for {chunk_id}",
        page_number=1,
        section="Safety",
        score=0.5,
    )


class FakeEmbeddingProvider:
    def __init__(self, embeddings=None):
        self.embeddings = (
            [[1.0, 0.0, 0.0]]
            if embeddings is None
            else embeddings
        )
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(texts)
        return self.embeddings


class FakeVectorStore:
    def __init__(self, results=None):
        self.results = results or []
        self.queries = []

    def search(self, query_embedding, top_k=5):
        self.queries.append((query_embedding, top_k))
        return self.results[:top_k]


class FakeBM25Retriever:
    def __init__(self, results=None):
        self.results = results or []
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.results[:top_k]


def test_hybrid_search_fuses_results_from_both_retrievers():
    vector_store = FakeVectorStore(
        [make_result("shared"), make_result("semantic-only")]
    )
    bm25 = FakeBM25Retriever(
        [make_result("shared"), make_result("lexical-only")]
    )
    provider = FakeEmbeddingProvider()

    service = HybridRetrievalService(
        vector_store,
        bm25,
        provider,
        candidate_k=10,
    )

    results = service.search("workplace safety")

    assert results[0].chunk_id == "shared"
    assert {result.chunk_id for result in results} == {
        "shared",
        "semantic-only",
        "lexical-only",
    }
    assert provider.calls == [["workplace safety"]]
    assert vector_store.queries == [([1.0, 0.0, 0.0], 10)]
    assert bm25.queries == [("workplace safety", 10)]


def test_hybrid_search_respects_top_k():
    service = HybridRetrievalService(
        FakeVectorStore([make_result("a"), make_result("b")]),
        FakeBM25Retriever([make_result("c"), make_result("d")]),
        FakeEmbeddingProvider(),
    )

    assert len(service.search("safety", top_k=2)) == 2


def test_hybrid_search_returns_empty_when_no_results():
    service = HybridRetrievalService(
        FakeVectorStore(),
        FakeBM25Retriever(),
        FakeEmbeddingProvider(),
    )

    assert service.search("unmatched query") == []


def test_hybrid_search_rejects_blank_query():
    service = HybridRetrievalService(
        FakeVectorStore(),
        FakeBM25Retriever(),
        FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="query"):
        service.search("   ")


def test_hybrid_search_rejects_invalid_top_k():
    service = HybridRetrievalService(
        FakeVectorStore(),
        FakeBM25Retriever(),
        FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="top_k"):
        service.search("safety", top_k=0)


def test_hybrid_search_rejects_invalid_candidate_k():
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetrievalService(
            FakeVectorStore(),
            FakeBM25Retriever(),
            FakeEmbeddingProvider(),
            candidate_k=0,
        )


def test_hybrid_search_rejects_missing_query_embedding():
    service = HybridRetrievalService(
        FakeVectorStore(),
        FakeBM25Retriever(),
        FakeEmbeddingProvider(embeddings=[]),
    )

    with pytest.raises(ValueError, match="exactly one"):
        service.search("safety")
