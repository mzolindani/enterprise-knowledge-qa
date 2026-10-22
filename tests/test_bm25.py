
import pytest

from app.chunking.models import DocumentChunk
from app.retrieval.bm25_store import BM25Retriever


def make_chunk(
    chunk_id: str,
    text: str,
    document_id: str = "document-001",
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        content_hash="test-hash",
        chunk_index=chunk_index,
        page_number=1,
        section="Safety",
        text=text,
        token_count=len(text.split()),
    )


@pytest.fixture
def retriever():
    instance = BM25Retriever()
    instance.index_chunks(
        [
            make_chunk(
                "chunk-safety",
                "Employees must report workplace safety incidents.",
                chunk_index=0,
            ),
            make_chunk(
                "chunk-stock",
                "Inventory stock levels must be checked before fulfilment.",
                chunk_index=1,
            ),
            make_chunk(
                "chunk-reporting",
                "Incident reporting must include the date and manager.",
                chunk_index=2,
            ),
        ]
    )
    return instance


def test_index_chunks_and_count(retriever):
    assert retriever.count() == 3


def test_search_returns_relevant_results(retriever):
    results = retriever.search("workplace safety incidents")

    assert results
    assert results[0].chunk_id == "chunk-safety"
    assert all(result.score > 0 for result in results)


def test_search_returns_existing_retrieval_result_fields(retriever):
    results = retriever.search("inventory stock")

    assert results
    result = results[0]

    assert result.document_id == "document-001"
    assert result.document_name == "document-001.pdf"
    assert result.page_number == 1
    assert result.section == "Safety"
    assert result.text


def test_search_respects_top_k(retriever):
    results = retriever.search("incident reporting safety", top_k=1)

    assert len(results) <= 1


def test_search_returns_empty_for_unmatched_terms(retriever):
    results = retriever.search("xylophone quasar")

    assert results == []


def test_search_returns_empty_for_blank_query(retriever):
    assert retriever.search("   ") == []


def test_search_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("safety", top_k=0)


def test_index_chunks_rejects_empty_text():
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="Chunk text"):
        retriever.index_chunks(
            [make_chunk("chunk-empty", "   ")]
        )


def test_index_chunks_rejects_duplicate_ids():
    retriever = BM25Retriever()
    chunks = [
        make_chunk("duplicate", "workplace safety"),
        make_chunk("duplicate", "inventory stock"),
    ]

    with pytest.raises(ValueError, match="unique"):
        retriever.index_chunks(chunks)


def test_replace_document_removes_obsolete_chunks():
    retriever = BM25Retriever()
    retriever.index_chunks(
        [
            make_chunk("old-1", "workplace safety", "document-001"),
            make_chunk("keep-1", "inventory stock", "document-002"),
        ]
    )

    retriever.replace_document(
        "document-001",
        [
            make_chunk(
                "new-1",
                "incident reporting",
                "document-001",
            )
        ],
    )

    assert retriever.count() == 2
    assert retriever.search("workplace safety") == []
    assert retriever.search("incident reporting")[0].chunk_id == "new-1"
    assert retriever.search("inventory stock")[0].chunk_id == "keep-1"


def test_replace_document_with_empty_chunks_removes_document():
    retriever = BM25Retriever()
    retriever.index_chunks(
        [
            make_chunk("chunk-1", "workplace safety", "document-001"),
            make_chunk("chunk-2", "inventory stock", "document-002"),
        ]
    )

    retriever.replace_document("document-001", [])

    assert retriever.count() == 1
    assert retriever.search("workplace safety") == []
    assert retriever.search("inventory stock")[0].chunk_id == "chunk-2"


def test_replace_document_rejects_chunks_from_another_document():
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="specified document"):
        retriever.replace_document(
            "document-001",
            [make_chunk("chunk-2", "inventory stock", "document-002")],
        )


def test_replace_document_rejects_empty_document_id():
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="document_id"):
        retriever.replace_document(" ", [])
