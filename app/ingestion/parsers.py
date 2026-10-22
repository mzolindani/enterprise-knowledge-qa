
import re
from io import BytesIO

import pymupdf
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingestion.models import PageContent


def parse_pdf(file_bytes: bytes) -> list[PageContent]:
    """Extract text while preserving original PDF page numbers."""

    pages: list[PageContent] = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    PageContent(
                        page_number=index,
                        text=text,
                    )
                )

    return pages


def parse_docx(file_bytes: bytes) -> list[PageContent]:
    """Extract paragraphs and tables in document-body order.

    DOCX physical page numbers are not assigned because they depend
    on document rendering and pagination settings.
    """

    document = Document(BytesIO(file_bytes))
    blocks: list[str] = []

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()

            if text:
                style_name = paragraph.style.name if paragraph.style else ""

                if style_name.startswith("Heading"):
                    level_match = re.search(r"\d+", style_name)
                    level = int(level_match.group()) if level_match else 1
                    level = min(max(level, 1), 6)
                    blocks.append(f"{'#' * level} {text}")
                else:
                    blocks.append(text)
        elif isinstance(element, CT_Tbl):
            table = Table(element, document)
            rows: list[str] = []

            for row in table.rows:
                cells = [
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                ]
                rows.append(" | ".join(cells))

            table_text = "\n".join(rows).strip()

            if table_text:
                blocks.append(
                    "[TABLE]\n" + table_text + "\n[/TABLE]"
                )

    text = "\n\n".join(blocks).strip()

    if not text:
        return []

    return [
        PageContent(
            page_number=None,
            text=text,
        )
    ]
