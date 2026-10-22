
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceCitation:
    """Reference to a document chunk supporting an answer."""

    document_id: str
    document_name: str
    page_number: int | None
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class AnswerValidation:
    """Result of checking an answer against retrieved evidence."""

    valid: bool
    supported_claims: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Answer generated from retrieved document evidence."""

    question: str
    answer: str
    citations: tuple[SourceCitation, ...]
    model: str
    grounded: bool
