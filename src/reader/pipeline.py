import asyncio

from .config import Settings
from .sessions import Result, Session
from .textproc import chunk_text, guess_title, merge_pages
from .tts import Tts, synthesize_document


class EmptyDocument(RuntimeError):
    pass


async def wait_for_pages(session: Session, tasks: list[asyncio.Task]) -> None:
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def collect_text(session: Session) -> tuple[str, str]:
    texts = [page.text for page in session.pages if page.status == "ready" and page.text]
    if not texts:
        raise EmptyDocument("no readable pages in this batch")
    merged = merge_pages(texts)
    language = next(
        (page.language for page in session.pages if page.status == "ready" and page.language),
        None,
    )
    return merged, language or "english"


async def finalize_session(session: Session, tts: Tts, settings: Settings) -> Result:
    text, language = collect_text(session)
    title = guess_title(text) or "Recording"
    chunks = chunk_text(text, settings.tts_chunk_chars)
    audio_paths = await synthesize_document(
        chunks,
        title,
        language,
        session.dir,
        tts,
        part_chars=settings.tts_part_chars,
        concurrency=settings.tts_concurrency,
        bitrate=settings.audio_bitrate,
    )
    transcript = session.dir / "transcript.txt"
    transcript.write_text(text, encoding="utf-8")
    return Result(
        title=title,
        language=language,
        text=text,
        audio_paths=[str(path) for path in audio_paths],
        transcript_path=str(transcript),
    )
