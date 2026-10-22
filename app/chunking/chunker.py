
import hashlib  # noqa: I001
import re
from dataclasses import dataclass

import tiktoken

from app.chunking.models import DocumentChunk
from app.ingestion.models import IngestedDocument


DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50

_ENCODING = tiktoken.get_encoding("cl100k_base")

_TABLE_PATTERN = re.compile(
    r"\[TABLE\]\s*(.*?)\s*\[/TABLE\]",
    flags=re.DOTALL | re.IGNORECASE,
)
_HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class _Block:
    text: str
    section: str | None
    kind: str = "text"


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_by_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Split text at word boundaries while respecting token limits."""
    words = text.split()

    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start
        current_words: list[str] = []

        while end < len(words):
            candidate = current_words + [words[end]]
            candidate_text = " ".join(candidate)

            if _count_tokens(candidate_text) > size:
                break

            current_words.append(words[end])
            end += 1

        if not current_words:
            raise ValueError(
                "A single word exceeds chunk_size. "
                "Increase chunk_size or preprocess the text."
            )

        chunks.append(" ".join(current_words))

        if end >= len(words):
            break

        # Retain complete trailing words within the overlap token budget.
        next_start = end
        

        for candidate_start in range(end - 1, start - 1, -1):
            candidate_text = " ".join(words[candidate_start:end])

            if _count_tokens(candidate_text) > overlap:
                break

            next_start = candidate_text
            next_start = candidate_start

        # Guarantee progress, even if no complete word fits the overlap.
        start = min(end, next_start)

    return chunks


def _extract_blocks(text: str) -> list[_Block]:
    """Recognise headings, paragraphs and explicitly marked tables."""
    blocks: list[_Block] = []
    current_section: str | None = None

    # Replace tables with placeholders so their rows can be handled
    # separately from ordinary paragraphs.
    tables: list[str] = []

    def save_table(match: re.Match[str]) -> str:
        tables.append(match.group(1).strip())
        return f"\n\n[[TABLE_BLOCK_{len(tables) - 1}]]\n\n"

    text_with_placeholders = _TABLE_PATTERN.sub(save_table, text)

    for paragraph in re.split(r"\n\s*\n", text_with_placeholders):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        table_match = re.fullmatch(
            r"\[\[TABLE_BLOCK_(\d+)\]\]", paragraph
        )

        if table_match:
            table_text = tables[int(table_match.group(1))]
            rows = [
                line.strip()
                for line in table_text.splitlines()
                if line.strip()
            ]

            if rows:
                blocks.append(
                    _Block(
                        text="\n".join(rows),
                        section=current_section,
                        kind="table",
                    )
                )
            continue

        heading_match = _HEADING_PATTERN.match(paragraph)

        if heading_match:
            current_section = heading_match.group(2).strip()
            blocks.append(
                _Block(
                    text=paragraph,
                    section=current_section,
                    kind="heading",
                )
            )
        else:
            blocks.append(
                _Block(
                    text=paragraph,
                    section=current_section,
                )
            )

    return blocks


def _split_table(
    table_text: str,
    chunk_size: int,
) -> list[str]:
    """Split a table into row groups and repeat its header."""
    rows = [row for row in table_text.splitlines() if row.strip()]

    if not rows:
        return []

    header = rows[0]
    data_rows = rows[1:]

    # The header must fit by itself for the repeated-header guarantee.
    if _count_tokens(header) > chunk_size:
        raise ValueError(
            "Table header exceeds chunk_size; shorten the header "
            "or increase the configured chunk size."
        )

    chunks: list[str] = []
    current_rows: list[str] = [header]

    for row in data_rows:
        candidate = "\n".join(current_rows + [row])

        if _count_tokens(candidate) <= chunk_size:
            current_rows.append(row)
            continue

        if len(current_rows) == 1:
            # A single data row is too large to fit with its header.
            row_chunks = _split_by_tokens(row, chunk_size, 0)

            for row_chunk in row_chunks:
                combined = f"{header}\n{row_chunk}"

                # If even the header plus a row fragment is too large,
                # fail explicitly instead of silently exceeding limits.
                if _count_tokens(combined) > chunk_size:
                    raise ValueError(
                        "A table row cannot fit with its header. "
                        "Increase chunk_size or preprocess this table."
                    )

                chunks.append(combined)

            current_rows = [header]
            continue

        chunks.append("\n".join(current_rows))
        current_rows = [header, row]

        if _count_tokens("\n".join(current_rows)) > chunk_size:
            row_chunks = _split_by_tokens(row, chunk_size, 0)
            current_rows = [header]

            for row_chunk in row_chunks:
                combined = f"{header}\n{row_chunk}"

                if _count_tokens(combined) > chunk_size:
                    raise ValueError(
                        "A table row cannot fit with its header. "
                        "Increase chunk_size or preprocess this table."
                    )

                chunks.append(combined)

    if len(current_rows) > 1 or not chunks:
        chunks.append("\n".join(current_rows))

    return chunks


def _pack_text_blocks(
    blocks: list[_Block],
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, str | None]]:
    """Pack paragraph blocks while retaining context between chunks."""
    results: list[tuple[str, str | None]] = []
    current_text = ""
    current_section: str | None = None

    def emit(text: str, section: str | None) -> None:
        if text.strip():
            results.append((text.strip(), section))

    for block in blocks:
        text = block.text.strip()

        if block.kind == "heading":
            if current_text:
                emit(current_text, current_section)
                current_text = ""

            current_section = block.section
            current_text = text
            continue

        if block.kind == "table":
            if current_text:
                emit(current_text, current_section)
                current_text = ""

            for table_chunk in _split_table(text, chunk_size):
                emit(table_chunk, block.section)

            continue

        if _count_tokens(text) > chunk_size:
            if current_text:
                emit(current_text, current_section)
                current_text = ""

            long_chunks = _split_by_tokens(text, chunk_size, overlap)

            for index, long_chunk in enumerate(long_chunks):
                if index == 0:
                    emit(long_chunk, block.section)
                else:
                    emit(long_chunk, block.section)

            continue

        candidate = (
            f"{current_text}\n\n{text}" if current_text else text
        )

        if current_text and _count_tokens(candidate) > chunk_size:
            previous_text = current_text
            emit(previous_text, current_section)

            previous_tokens = _ENCODING.encode(previous_text)
            tail = _ENCODING.decode(previous_tokens[-overlap:]) if overlap else ""

            current_text = (
                f"{tail}\n\n{text}" if tail else text
            )

            # If overlap pushes the new chunk over the limit, remove it.
            if _count_tokens(current_text) > chunk_size:
                current_text = text

            current_section = block.section or current_section
        else:
            current_text = candidate
            current_section = block.section or current_section

    if current_text:
        emit(current_text, current_section)

    return results


def chunk_document(
    document: IngestedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Create token-limited, traceable chunks from an ingested document.

    Token counts use cl100k_base. PDF page numbers are preserved.
    DOCX documents currently have no physical page numbers.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    global_index = 0

    for page in document.pages:
        normalised = _normalise_text(page.text)

        if not normalised:
            continue

        blocks = _extract_blocks(normalised)
        # text_blocks = [
        #     block for block in blocks if block.kind != "table"
        # ]
        # table_blocks = [
        #     block for block in blocks if block.kind == "table"
        # ]

        # Process in original order so sections and tables remain
        # attached to the appropriate source context.
        ordered_results: list[tuple[str, str | None]] = []
        pending_text: list[_Block] = []

        for block in blocks:
            if block.kind != "table":
                pending_text.append(block)
                continue

            if pending_text:
                ordered_results.extend(
                    _pack_text_blocks(pending_text, chunk_size, overlap)
                )
                pending_text = []

            ordered_results.extend(
                (table_chunk, block.section)
                for table_chunk in _split_table(block.text, chunk_size)
            )

        if pending_text:
            ordered_results.extend(
                _pack_text_blocks(pending_text, chunk_size, overlap)
            )

        for text, section in ordered_results:
            text = text.strip()

            if not text:
                continue

            token_count = _count_tokens(text)

            if token_count > chunk_size:
                raise ValueError(
                    f"Chunk exceeds configured token limit: "
                    f"{token_count} > {chunk_size}"
                )

            chunk_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            identity = (
                f"{document.document_id}:"
                f"{page.page_number}:"
                f"{global_index}:"
                f"{chunk_hash}"
            )
            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_name=document.document_name,
                    content_hash=chunk_hash,
                    chunk_index=global_index,
                    page_number=page.page_number,
                    section=section,
                    text=text,
                    token_count=token_count,
                )
            )

            global_index += 1

    return chunks
