
import math
import uuid

import pytest

from app.chunking.models import DocumentChunk
from app.embeddings.models import EmbeddedChunk
from app.retrieval import ChromaVectorStore


@pytest.fixture
def store():


    collection_name = f"test_{uuid.uuid4().hex}"
    vector_store = ChromaVectorStore(
        collection_name=collection_name
    )

    yield vector_store

    vector_store.client.delete_collection(
        name=collection_name
    )


def make_embedded_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    chunk_index: int,
    page_number: int | None = 1,
    section: str | None = "Safety",
    document_id: str = "document-001",
) -> EmbeddedChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name="safety-policy.pdf",
        content_hash="test-hash",
        chunk_index=chunk_index,
        page_number=page_number,
        section=section,
        text=text,
        token_count=len(text.split()),
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=tuple(embedding),
    )

def test_add_chunks_stores_embeddings_and_metadata(store):
    item = make_embedded_chunk(
        "chunk-001",
        "Report workplace safety incidents.",
        [1.0, 0.0, 0.0],
        0,
        page_number=3,
        section="Incident Reporting",
    )

    store.add_chunks([item])

    assert store.count() == 1

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-001"
    assert results[0].document_id == "document-001"
    assert results[0].document_name == "safety-policy.pdf"
    assert results[0].page_number == 3
    assert results[0].section == "Incident Reporting"
    assert results[0].text == "Report workplace safety incidents."


def test_search_ranks_most_similar_chunk_first(store):
    items = [
        make_embedded_chunk(
            "chunk-x",
            "Unrelated content.",
            [0.0, 1.0, 0.0],
            0,
        ),
        make_embedded_chunk(
            "chunk-y",
            "Relevant safety content.",
            [1.0, 0.0, 0.0],
            1,
        ),
    ]

    store.add_chunks(items)

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert results[0].chunk_id == "chunk-y"
    assert results[0].score > results[1].score
    assert results[0].score == pytest.approx(1.0)


def test_search_limits_results_to_top_k(store):
    items = [
        make_embedded_chunk(
            f"chunk-{index}",
            f"Document text {index}.",
            [1.0, float(index + 1), 0.0],
            index,
        )
        for index in range(4)
    ]

    store.add_chunks(items)

    results = store.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_returns_empty_for_empty_collection(store):
    assert store.search([1.0, 0.0, 0.0]) == []


def test_add_empty_list_does_nothing(store):
    store.add_chunks([])

    assert store.count() == 0


def test_upsert_does_not_duplicate_chunk_ids(store):
    item = make_embedded_chunk(
        "chunk-001",
        "Original text.",
        [1.0, 0.0, 0.0],
        0,
    )

    store.add_chunks([item])
    store.add_chunks([item])

    assert store.count() == 1


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_invalid_top_k(store, top_k):
    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0, 0.0], top_k=top_k)


def test_search_rejects_empty_query_embedding(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store.search([])


def test_search_rejects_non_finite_query_embedding(store):
    with pytest.raises(ValueError, match="finite numeric"):
        store.search([math.nan, 0.0, 1.0])


def test_clear_removes_all_chunks(store):
    store.add_chunks([
        make_embedded_chunk(
            "chunk-001",
            "First text.",
            [1.0, 0.0, 0.0],
            0,
        ),
        make_embedded_chunk(
            "chunk-002",
            "Second text.",
            [0.0, 1.0, 0.0],
            1,
        ),
    ])

    store.clear()

    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0]) == []

def test_persistent_store_retains_chunks_after_reopening(tmp_path):
    persist_directory = tmp_path / "chroma_data"
    collection_name = "persistence_test"

    first_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    item = make_embedded_chunk(
        "persistent-chunk-001",
        "Employees must report workplace incidents.",
        [1.0, 0.0, 0.0],
        0,
        page_number=2,
        section="Incident Reporting",
    )

    first_store.add_chunks([item])

    # Open a new client against the same persistent directory.
    second_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    results = second_store.search([1.0, 0.0, 0.0], top_k=1)

    assert second_store.count() == 1
    assert len(results) == 1
    assert results[0].chunk_id == "persistent-chunk-001"
    assert results[0].page_number == 2
    assert results[0].section == "Incident Reporting"


def test_replace_document_removes_obsolete_chunks(store):
    old_chunks = [
        make_embedded_chunk(
            "old-chunk-001",
            "Old safety policy.",
            [1.0, 0.0, 0.0],
            0,
        ),
        make_embedded_chunk(
            "old-chunk-002",
            "Obsolete reporting procedure.",
            [0.0, 1.0, 0.0],
            1,
        ),
    ]

    store.add_chunks(old_chunks)

    replacement = make_embedded_chunk(
        "new-chunk-001",
        "Updated safety policy.",
        [1.0, 0.0, 0.0],
        0,
    )

    store.replace_document(
        "document-001",
        [replacement],
    )

    assert store.count() == 1
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert [result.chunk_id for result in results] == ["new-chunk-001"]


def test_replace_document_with_empty_chunks_removes_document(store):
    store.add_chunks([
        make_embedded_chunk(
            "old-chunk-001",
            "Old safety policy.",
            [1.0, 0.0, 0.0],
            0,
        ),
    ])

    store.replace_document("document-001", [])

    assert store.count() == 0


def test_replace_document_rejects_chunks_from_another_document(store):
    wrong_document_chunk = make_embedded_chunk(
        "other-chunk-001",
        "Different document.",
        [1.0, 0.0, 0.0],
        0,
        document_id="document-002",
    )

    assert wrong_document_chunk.document_id == "document-002"

    with pytest.raises(ValueError, match="must belong"):
        store.replace_document(
            "document-001",
            [wrong_document_chunk],
        )

    assert store.count() == 0
