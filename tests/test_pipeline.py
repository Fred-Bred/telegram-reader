import asyncio
from pathlib import Path

import pytest

import reader.tts as tts_module
from reader.config import Settings
from reader.pipeline import EmptyDocument, collect_text, finalize_session
from reader.sessions import Page, Session
from reader.textproc import chunk_text


class FakeTts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text: str, language: str) -> bytes:
        self.calls.append((text, language))
        return f"AUDIO:{text}".encode()


@pytest.fixture
def fake_concat(monkeypatch):
    async def join(chunk_paths, out_path, bitrate="48k"):
        out_path.write_bytes(b"".join(p.read_bytes() for p in sorted(chunk_paths)))

    monkeypatch.setattr(tts_module, "concat_mp3", join)


def make_settings(tmp_path, **kwargs) -> Settings:
    return Settings(telegram_bot_token="1:t", openai_api_key="k", data_dir=tmp_path, **kwargs)


def make_session(tmp_path) -> Session:
    session = Session.new(1, tmp_path)
    session.pages = [
        Page(
            index=1,
            kind="image",
            source="a",
            status="ready",
            text="Page one ends with expe-",
            language="danish",
        ),
        Page(
            index=2,
            kind="image",
            source="b",
            status="ready",
            text="rience of research.",
            language="danish",
        ),
    ]
    return session


def test_collect_text_raises_when_nothing_ready(tmp_path):
    session = make_session(tmp_path)
    session.pages[0].status = "unreadable"
    session.pages[1].status = "failed"
    with pytest.raises(EmptyDocument):
        collect_text(session)


async def test_finalize_session_produces_result(fake_concat, tmp_path):
    settings = make_settings(tmp_path)
    session = make_session(tmp_path)
    tts = FakeTts()

    result = await finalize_session(session, tts, settings)

    assert result.title
    assert result.language == "danish"
    assert result.text == "Page one ends with experience of research."
    assert result.audio_paths
    for audio in result.audio_paths:
        assert Path(audio).is_file()
    assert Path(result.transcript_path).read_text(encoding="utf-8") == result.text
    assert tts.calls
    for text, language in tts.calls:
        assert len(text) <= settings.tts_chunk_chars
        assert language == "danish"


async def test_chunking_used_for_long_document(fake_concat, tmp_path):
    settings = make_settings(tmp_path, tts_chunk_chars=50)
    session = make_session(tmp_path)
    long_text = "Sentence here. " * 20
    session.pages = [
        Page(
            index=1,
            kind="pdf_text",
            source=long_text,
            status="ready",
            text=long_text,
            language="english",
        )
    ]
    tts = FakeTts()

    result = await finalize_session(session, tts, settings)

    assert len(chunk_text(long_text, 50)) > 1
    assert len(result.audio_paths) == 1
    content = Path(result.audio_paths[0]).read_bytes()
    assert b"Sentence here" in content


def test_grouped_parts_when_many_chunks(fake_concat, tmp_path):
    settings = make_settings(tmp_path, tts_chunk_chars=10, tts_part_chars=12)
    session = make_session(tmp_path)
    text = "one two three four five six seven eight nine ten"
    session.pages = [
        Page(index=1, kind="pdf_text", source=text, status="ready", text=text, language="english")
    ]
    tts = FakeTts()

    result = asyncio.run(finalize_session(session, tts, settings))

    assert len(result.audio_paths) > 1
    assert all(Path(p).is_file() for p in result.audio_paths)
