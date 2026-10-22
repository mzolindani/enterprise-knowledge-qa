
import hashlib
import os
from pathlib import Path

from app.ingestion.models import IngestedDocument
from app.ingestion.parsers import parse_docx, parse_pdf

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def ingest_file(file_path: str | Path) -> IngestedDocument:
    """Read a supported document and return structured content."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension or '(no extension)'}. "
            "Supported types: PDF, DOCX."
        )

    file_bytes = path.read_bytes()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    canonical_path = os.path.normcase(
        str(path.resolve())
    )

    document_id = hashlib.sha256(
        canonical_path.encode("utf-8")
    ).hexdigest()
    try:
        if extension == ".pdf":
            pages = parse_pdf(file_bytes)
        else:
            pages = parse_docx(file_bytes)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise

        raise ValueError(
            f"Could not parse document '{path.name}': {exc}"
        ) from exc

    if not pages or not any(page.text.strip() for page in pages):
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The document may be empty or require OCR."
        )

    return IngestedDocument(
        document_id=document_id,
        document_name=path.name,
        file_type=extension.lstrip("."),
        content_hash=content_hash,
        file_size_bytes=len(file_bytes),
        source_path=str(path.resolve()),
        pages=tuple(pages),
    )
