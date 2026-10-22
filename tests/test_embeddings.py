
import numpy as np
import pytest

from app.chunking import chunk_document
from app.embeddings import (
    EmbeddedChunk,
    LocalSentenceTransformerProvider,
    embed_chunks,
)
from app.ingestion import ingest_file


class FakeSentenceEncoder:
    """Fake encoder that returns deterministic vectors."""

    def __init__(self):
        self.batches = []

    def encode(
        self,
        texts,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        self.batches.append(list(texts))
        return np.array([
            [3.0, 4.0, 0.0] for _ in texts
        ])


class FakeEmbeddingProvider:
    """Deterministic provider for tests without API calls."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.received_texts: list[str] = []

    def embed_texts(self, texts):
        self.received_texts.extend(texts)
        return [
            [float(index + 1)] * self.dimension
            for index, _ in enumerate(texts)
        ]


def make_chunks(tmp_path):
    import pymupdf

    path = tmp_path / "embedding-test.pdf"

    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (50, 50, 550, 750),
            "The incident reporting procedure explains "
            "how employees report workplace safety incidents.",
            fontsize=10,
        )
        pdf.save(path)

    document = ingest_file(path)
    return chunk_document(document)


def test_local_provider_batches_texts():
    encoder = FakeSentenceEncoder()
    provider = LocalSentenceTransformerProvider(
        batch_size=2,
        encoder=encoder,
    )

    results = provider.embed_texts(["first", "second", "third"])

    assert len(results) == 3
    assert [len(batch) for batch in encoder.batches] == [2, 1]


def test_local_provider_normalises_embeddings():
    provider = LocalSentenceTransformerProvider(
        encoder=FakeSentenceEncoder(),
    )

    vector = provider.embed_texts(["example"])[0]

    assert vector == pytest.approx([0.6, 0.8, 0.0])


def test_local_provider_rejects_empty_text():
    provider = LocalSentenceTransformerProvider(
        encoder=FakeSentenceEncoder(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        provider.embed_texts(["valid text", " "])


def test_local_provider_returns_empty_for_empty_input():
    encoder = FakeSentenceEncoder()
    provider = LocalSentenceTransformerProvider(encoder=encoder)

    assert provider.embed_texts([]) == []
    assert encoder.batches == []


def test_embed_chunks_preserves_source_metadata(tmp_path):
    chunks = make_chunks(tmp_path)
    provider = FakeEmbeddingProvider()

    results = embed_chunks(chunks, provider)

    assert len(results) == len(chunks)
    assert isinstance(results[0], EmbeddedChunk)
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].document_id == chunks[0].document_id
    assert results[0].document_name == chunks[0].document_name
    assert results[0].page_number == chunks[0].page_number
    assert len(results[0].embedding) == 3


def test_embed_chunks_passes_chunk_text_to_provider(tmp_path):
    chunks = make_chunks(tmp_path)
    provider = FakeEmbeddingProvider()

    embed_chunks(chunks, provider)

    assert provider.received_texts == [
        chunk.text for chunk in chunks
    ]


def test_embed_chunks_returns_empty_list_for_no_chunks():
    provider = FakeEmbeddingProvider()

    assert embed_chunks([], provider) == []
    assert provider.received_texts == []


def test_embed_chunks_rejects_wrong_vector_count(tmp_path):
    chunks = make_chunks(tmp_path)

    class WrongCountProvider:
        def embed_texts(self, texts):
            return []

    with pytest.raises(ValueError, match="count"):
        embed_chunks(chunks, WrongCountProvider())


def test_embed_chunks_rejects_inconsistent_dimensions(tmp_path):
    chunks = make_chunks(tmp_path)

    # Duplicate a chunk so the test has at least two inputs.
    chunks = chunks * 2

    class InconsistentProvider:
        def embed_texts(self, texts):
            return [
                [0.1, 0.2] if index == 0 else [0.3, 0.4, 0.5]
                for index, _ in enumerate(texts)
            ]

    with pytest.raises(
        ValueError,
        match="same non-zero dimension",
    ):
        embed_chunks(chunks, InconsistentProvider())


def test_embed_chunks_rejects_non_finite_values(tmp_path):
    chunks = make_chunks(tmp_path)

    class InvalidValueProvider:
        def embed_texts(self, texts):
            return [
                [float("nan"), 0.2, 0.3]
                for _ in texts
            ]

    with pytest.raises(ValueError, match="Invalid embedding values"):
        embed_chunks(chunks, InvalidValueProvider())
