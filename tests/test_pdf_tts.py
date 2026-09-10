import asyncio

import pymupdf

from reader.pdf import load_pdf_pages
from reader.tts import concat_mp3


async def _make_mp3(path, freq: float) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency={freq}:duration=0.3",
        "-c:a",
        "libmp3lame",
        str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()


async def test_concat_mp3_joins_real_files(tmp_path):
    first = tmp_path / "a.mp3"
    second = tmp_path / "b.mp3"
    await _make_mp3(first, 200)
    await _make_mp3(second, 300)
    out = tmp_path / "joined.mp3"
    await concat_mp3([first, second], out, "48k")
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_load_pdf_text_layer(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    lines = [
        "Line one of the quick brown fox story explains the fox.",
        "Line two of the story continues to explain the lazy dog.",
        "Line three of the story concludes with a summary sentence.",
    ]
    for i, line in enumerate(lines):
        page.insert_text((72, 100 + i * 20), line)
    document.save(pdf_path)

    pages = load_pdf_pages(pdf_path, tmp_path / "out")
    assert len(pages) == 1
    assert pages[0].text is not None and "quick brown fox" in pages[0].text
    assert pages[0].image_path is None


def test_load_pdf_blank_page_renders_image(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)

    pages = load_pdf_pages(pdf_path, tmp_path / "out")
    assert len(pages) == 1
    assert pages[0].text is None
    assert pages[0].image_path is not None and pages[0].image_path.is_file()
