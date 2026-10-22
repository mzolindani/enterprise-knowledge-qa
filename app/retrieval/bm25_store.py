
import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.chunking.models import DocumentChunk
from app.retrieval.models import RetrievalResult


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

def has_term_overlap(
    query_tokens: list[str],
    document_tokens: list[str],
) -> bool:
    """Return whether the query and document share at least one term."""
    return bool(set(query_tokens) & set(document_tokens))

class BM25Retriever:
    """In-memory lexical retriever using BM25Okapi."""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._index: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace the lexical index with the supplied chunks."""
        chunk_list = list(chunks)

        if any(not chunk.text.strip() for chunk in chunk_list):
            raise ValueError("Chunk text cannot be empty.")

        ids = [chunk.chunk_id for chunk in chunk_list]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk IDs must be unique.")

        self._chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
        self._chunk_ids = ids

        if not chunk_list:
            self._index = None
            return

        tokenized_corpus = [
            tokenize(chunk.text)
            for chunk in chunk_list
        ]

        self._index = BM25Okapi(tokenized_corpus)


   
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the highest-ranked lexical matches."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)
        if not query_tokens or self._index is None:
            return []

        scores = self._index.get_scores(query_tokens)

        matching_indices = [
            index
            for index, chunk_id in enumerate(self._chunk_ids)
            if has_term_overlap(
                query_tokens,
                tokenize(self._chunks[chunk_id].text),
            )
        ]

        ranked_indices = sorted(
            matching_indices,
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            chunk = self._chunks[self._chunk_ids[index]]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    score=float(scores[index]),
                )
            )

        return results


    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def replace_document(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        """Replace all chunks belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(
                "All chunks must belong to the specified document."
            )

        retained = [
            chunk
            for chunk in self._chunks.values()
            if chunk.document_id != document_id
        ]

        self.index_chunks([*retained, *chunks])
