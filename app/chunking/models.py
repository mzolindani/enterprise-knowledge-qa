
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable passage linked to its original document."""

    chunk_id: str
    document_id: str
    document_name: str
    content_hash: str
    chunk_index: int
    page_number: int | None
    section: str | None
    text: str
    token_count: int
