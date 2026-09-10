import asyncio
import shutil
from pathlib import Path
from typing import Protocol

from openai import AsyncOpenAI

from .textproc import group_chunks, sanitize_filename


class AudioError(RuntimeError):
    pass


class Tts(Protocol):
    async def synthesize(self, text: str, language: str) -> bytes: ...


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError("ffmpeg was not found on PATH")
    return path


def _instructions(language: str) -> str:
    return (
        f"Read the text aloud in {language} for someone listening to an academic text. "
        f"Use a calm, even, unhurried pace with clear, natural {language} pronunciation. "
        "Pause briefly between paragraphs."
    )


class OpenAiTts:
    def __init__(self, client: AsyncOpenAI, model: str, voice: str) -> None:
        self._client = client
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str, language: str) -> bytes:
        speech = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            instructions=_instructions(language),
            response_format="mp3",
        )
        return speech.content


async def concat_mp3(chunk_paths: list[Path], out_path: Path, bitrate: str = "48k") -> None:
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "".join(f"file '{chunk.resolve()}'\n" for chunk in chunk_paths), encoding="utf-8"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-ac",
        "1",
        str(out_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    list_path.unlink(missing_ok=True)
    if process.returncode != 0:
        detail = stderr.decode(errors="replace")[-400:]
        raise AudioError(f"ffmpeg failed: {detail}")


async def synthesize_document(
    chunks: list[str],
    title: str,
    language: str,
    out_dir: Path,
    tts: Tts,
    *,
    part_chars: int,
    concurrency: int,
    bitrate: str = "48k",
) -> list[Path]:
    groups = group_chunks(chunks, part_chars)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    flat: list[tuple[int, int, str]] = []
    for group_index, group in enumerate(groups):
        for chunk_index, chunk in enumerate(group):
            flat.append((group_index, chunk_index, chunk))

    async def one(group_index: int, chunk_index: int, chunk: str) -> Path:
        async with semaphore:
            audio = await tts.synthesize(chunk, language)
        path = chunk_dir / f"{group_index:02d}_{chunk_index:04d}.mp3"
        path.write_bytes(audio)
        return path

    synthesized = await asyncio.gather(*(one(*item) for item in flat))

    by_group: dict[int, list[Path]] = {}
    for (group_index, _, _), path in zip(flat, synthesized, strict=True):
        by_group.setdefault(group_index, []).append(path)

    slug = sanitize_filename(title)
    parts: list[Path] = []
    for group_index in sorted(by_group):
        out_name = f"{slug}.mp3"
        if len(groups) > 1:
            out_name = f"{slug} - part {group_index + 1:02d}.mp3"
        out_path = out_dir / out_name
        await concat_mp3(sorted(by_group[group_index]), out_path, bitrate)
        parts.append(out_path)
    return parts
