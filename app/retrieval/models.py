
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document chunk and its similarity information."""

    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_number: int | None
    section: str | None
    score: float
