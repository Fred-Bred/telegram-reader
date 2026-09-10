import argparse
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

from .config import get_settings
from .ocr import OpenAITextCleaner, OpenAIVisionReader
from .pdf import load_pdf_pages
from .textproc import chunk_text, guess_title, merge_pages
from .tts import OpenAiTts, require_ffmpeg, synthesize_document

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}


async def run(inputs: list[Path], out_dir: Path, voice: str | None) -> None:
    require_ffmpeg()
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    reader = OpenAIVisionReader(client, settings.ocr_model, settings.ocr_concurrency)
    cleaner = OpenAITextCleaner(client, settings.cleanup_model)
    tts = OpenAiTts(client, settings.tts_model, voice or settings.tts_voice)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_texts: list[str] = []
    language = "english"
    for item in inputs:
        if not item.is_file():
            raise SystemExit(f"not a file: {item}")
        if item.suffix.lower() == ".pdf":
            pdf_pages = await asyncio.to_thread(load_pdf_pages, item, out_dir)
            for pdf_page in pdf_pages:
                if pdf_page.text is not None:
                    result = await cleaner.clean(pdf_page.text)
                else:
                    result = await reader.read_page(pdf_page.image_path)
                if not result.unreadable and result.text:
                    page_texts.append(result.text)
                    language = result.language
                else:
                    print(f"warning: {item.name} pdf page {pdf_page.index + 1} unreadable")
            continue
        if item.suffix.lower() not in EXTENSIONS:
            raise SystemExit(f"unsupported file type: {item}")
        result = await reader.read_page(item)
        if result.unreadable or not result.text:
            print(f"warning: {item.name} unreadable")
            continue
        page_texts.append(result.text)
        language = result.language
        print(f"read {item.name} ({len(result.text)} chars)")

    merged = merge_pages(page_texts)
    if not merged:
        raise SystemExit("no readable pages")
    title = guess_title(merged) or "Recording"
    chunks = chunk_text(merged, settings.tts_chunk_chars)
    parts = await synthesize_document(
        chunks,
        title,
        language,
        out_dir,
        tts,
        part_chars=settings.tts_part_chars,
        concurrency=settings.tts_concurrency,
        bitrate=settings.audio_bitrate,
    )
    transcript = out_dir / "transcript.txt"
    transcript.write_text(merged, encoding="utf-8")
    print(f"\ntitle: {title}")
    print(f"language: {language}  chunks: {len(chunks)}")
    print(f"transcript: {transcript}")
    for part in parts:
        print(f"audio: {part}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reader", description="OCR images/PDFs and generate read-aloud MP3s locally"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="image files or PDFs")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory")
    parser.add_argument("--voice", default=None, help="TTS voice (default from settings)")
    args = parser.parse_args()
    asyncio.run(run(args.inputs, args.out, args.voice))


if __name__ == "__main__":
    main()
