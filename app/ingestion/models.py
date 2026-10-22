
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageContent:
    """A traceable unit of extracted document content."""

    page_number: int | None
    text: str


@dataclass(frozen=True)
class IngestedDocument:
    """Document metadata and its extracted content."""

    document_id: str
    document_name: str
    file_type: str
    content_hash: str
    file_size_bytes: int
    source_path: str
    pages: tuple[PageContent, ...]

    @property
    def full_text(self) -> str:
        """Combine extracted content without losing page boundaries."""
        return "\n\n".join(
            page.text for page in self.pages if page.text.strip()
        )


def get_document_name(path: str | Path) -> str:
    return Path(path).name
