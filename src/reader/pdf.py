from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class PdfPage:
    index: int
    text: str | None
    image_path: Path | None


def load_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    *,
    min_text_chars: int = 120,
    dpi: int = 170,
) -> list[PdfPage]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[PdfPage] = []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if len(text) >= min_text_chars:
                pages.append(PdfPage(index=index, text=text, image_path=None))
                continue
            pixmap = page.get_pixmap(dpi=dpi)
            image_path = out_dir / f"pdf_page_{index + 1:03d}.png"
            pixmap.save(image_path)
            pages.append(PdfPage(index=index, text=None, image_path=image_path))
    return pages
