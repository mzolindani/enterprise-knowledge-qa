
from dataclasses import dataclass

from app.chunking.models import DocumentChunk


@dataclass(frozen=True)
class EmbeddedChunk:
    """A document chunk paired with its semantic embedding."""

    chunk: DocumentChunk
    embedding: tuple[float, ...]

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def document_name(self) -> str:
        return self.chunk.document_name

    @property
    def page_number(self) -> int | None:
        return self.chunk.page_number

    @property
    def section(self) -> str | None:
        return self.chunk.section
