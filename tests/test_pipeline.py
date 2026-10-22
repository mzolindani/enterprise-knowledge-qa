
import pytest

from app.embeddings.models import EmbeddedChunk
from app.ingestion.pipeline import DocumentIndexingPipeline
from app.retrieval.bm25_store import BM25Retriever


class FakeEmbeddingProvider:
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def replace_document(self, document_id, chunks):
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.document_id != document_id
        ]

        self.chunks.extend(chunks)


def create_pdf(path):
    import pymupdf

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (50, 50, 550, 750),
            (
                "Workplace safety policy. Employees must report "
                "incidents to their manager and record the details "
                "in the incident reporting system."
            ),
            fontsize=12,
        )
        pdf.save(path)

def test_pipeline_indexes_document(tmp_path):
    pdf_path = tmp_path / "safety-policy.pdf"
    create_pdf(pdf_path)

    store = FakeVectorStore()
    pipeline = DocumentIndexingPipeline(
        vector_store=store,
        embedding_provider=FakeEmbeddingProvider(),
    )

    indexed_count = pipeline.index_file(pdf_path)

    assert indexed_count > 0
    assert len(store.chunks) == indexed_count
    assert all(
        isinstance(chunk, EmbeddedChunk)
        for chunk in store.chunks
    )
    assert store.chunks[0].document_name == "safety-policy.pdf"

def test_pipeline_rejects_unsupported_file(tmp_path):
    unsupported_file = tmp_path / "notes.txt"
    unsupported_file.write_text(
        "This file type is unsupported.",
        encoding="utf-8",
    )

    pipeline = DocumentIndexingPipeline(
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError):
        pipeline.index_file(unsupported_file)


def test_pipeline_does_not_embed_when_no_chunks_are_returned(
    tmp_path,
    monkeypatch,
):
    from app.ingestion import pipeline as pipeline_module

    class TrackingProvider:
        def __init__(self):
            self.called = False

        def embed_texts(self, texts):
            self.called = True
            return []

    provider = TrackingProvider()
    store = FakeVectorStore()

    monkeypatch.setattr(
        pipeline_module,
        "chunk_document",
        lambda document: [],
    )

    pipeline = DocumentIndexingPipeline(
        vector_store=store,
        embedding_provider=provider,
    )

    pdf_path = tmp_path / "empty-after-chunking.pdf"
    create_pdf(pdf_path)

    assert pipeline.index_file(pdf_path) == 0
    assert provider.called is False
    assert store.chunks == []
def test_pipeline_removes_old_chunks_when_reindexing_returns_no_chunks(
    tmp_path,
    monkeypatch,
):
    from app.ingestion import pipeline as pipeline_module

    pdf_path = tmp_path / "safety-policy.pdf"
    create_pdf(pdf_path)

    store = FakeVectorStore()
    provider = FakeEmbeddingProvider()
    pipeline = DocumentIndexingPipeline(
        vector_store=store,
        embedding_provider=provider,
    )

    # Index the document normally first.
    initial_count = pipeline.index_file(pdf_path)

    assert initial_count > 0
    assert len(store.chunks) == initial_count

    # Simulate a later indexing attempt that produces no chunks.
    monkeypatch.setattr(
        pipeline_module,
        "chunk_document",
        lambda document: [],
    )

    indexed_count = pipeline.index_file(pdf_path)

    assert indexed_count == 0
    assert store.chunks == []



def test_pipeline_updates_bm25_index(tmp_path):
    pdf_path = tmp_path / "safety-policy.pdf"
    create_pdf(pdf_path)

    vector_store = FakeVectorStore()
    bm25 = BM25Retriever()

    pipeline = DocumentIndexingPipeline(
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        bm25_retriever=bm25,
    )

    indexed_count = pipeline.index_file(pdf_path)

    assert indexed_count > 0
    assert len(vector_store.chunks) == indexed_count
    assert bm25.count() == indexed_count

    results = bm25.search("workplace safety", top_k=5)

    assert results
    assert results[0].document_name == "safety-policy.pdf"


def test_pipeline_removes_bm25_chunks_when_reindexing_returns_no_chunks(
    tmp_path,
    monkeypatch,
):
    from app.ingestion import pipeline as pipeline_module

    pdf_path = tmp_path / "safety-policy.pdf"
    create_pdf(pdf_path)

    vector_store = FakeVectorStore()
    bm25 = BM25Retriever()

    pipeline = DocumentIndexingPipeline(
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        bm25_retriever=bm25,
    )

    initial_count = pipeline.index_file(pdf_path)

    assert initial_count > 0
    assert bm25.count() == initial_count

    monkeypatch.setattr(
        pipeline_module,
        "chunk_document",
        lambda document: [],
    )

    indexed_count = pipeline.index_file(pdf_path)

    assert indexed_count == 0
    assert vector_store.chunks == []
    assert bm25.count() == 0
