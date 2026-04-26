from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz


@dataclass(slots=True)
class PageChunk:
    page_start: int
    page_end: int
    text: str


class PdfExtractor:
    def extract_text(self, pdf_path: str | Path) -> str:
        document = fitz.open(str(pdf_path))
        try:
            parts: list[str] = []
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                text = page.get_text("text")
                if text.strip():
                    parts.append(f"[Page {page_number + 1}]\n{text.strip()}")
            return "\n\n".join(parts)
        finally:
            document.close()

    def chunk_text(self, text: str, max_pages_per_chunk: int = 8) -> list[PageChunk]:
        pages = self._split_pages(text)
        chunks: list[PageChunk] = []
        for start in range(0, len(pages), max_pages_per_chunk):
            end = min(start + max_pages_per_chunk, len(pages))
            chunk_text = "\n\n".join(pages[start:end])
            chunks.append(PageChunk(page_start=start + 1, page_end=end, text=chunk_text))
        return chunks

    def _split_pages(self, text: str) -> list[str]:
        if not text.strip():
            return []
        pages = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("[Page ") and current:
                pages.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            pages.append("\n".join(current).strip())
        return [page for page in pages if page]

    def iter_form_widgets(self, pdf_path: str | Path) -> Iterable[dict[str, object]]:
        document = fitz.open(str(pdf_path))
        try:
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                for widget in page.widgets() or []:
                    yield {
                        "page": page_index + 1,
                        "field_name": widget.field_name or "",
                        "field_label": widget.field_label or "",
                        "field_type": widget.field_type or "",
                        "value": widget.field_value or "",
                        "rect": tuple(widget.rect),
                    }
        finally:
            document.close()
