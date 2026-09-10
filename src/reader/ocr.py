import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openai import AsyncOpenAI

from .textproc import chunk_text

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

OCR_SYSTEM_PROMPT = """\
You transcribe photographed pages from books and printed articles so the text can be read \
aloud by a speech engine.

Always answer with a JSON object: {"text": <string>, "language": <string>, "unreadable": <boolean>}.

Rules:
- Transcribe the main body of the page completely, in natural reading order. For multi-column \
layouts, read one column fully before starting the next.
- Keep the original language and wording exactly. Never translate, never summarize, never \
correct or improve the content, and never add commentary of your own.
- Merge words that are split across line ends, including words hyphenated at the line break.
- Preserve paragraph breaks. Do not merge separate paragraphs.
- Leave out running headers, footers, page numbers, signatures, library stamps and scan artifacts.
- If the page has a title or heading, transcribe it as the first line.
- Move footnotes and endnotes to the end of the text under a line reading "Notes:".
- Transcribe captions of figures and tables as part of the text.
- Set "language" to the language of the text as a lowercase English name, for example \
"english" or "danish".
- Set "unreadable" to true only if most of the page cannot be read. Still transcribe the \
parts that are legible.
"""

CLEANUP_SYSTEM_PROMPT = """\
You prepare extracted document text for a speech engine. The text comes from a PDF text layer \
and may contain broken hyphenation, repeated page headers, footers, page numbers, and \
scrambled column order.

Always answer with a JSON object: {"text": <string>, "language": <string>}.

Rules:
- Keep the original language and wording exactly. Never translate, never summarize, never \
add commentary of your own.
- Merge words hyphenated across line breaks.
- Remove running headers, footers and page numbers.
- Preserve paragraph breaks and natural reading order.
- Move footnotes and endnotes to the end of the text under a line reading "Notes:".
- Set "language" to the language of the text as a lowercase English name, for example \
"english" or "danish".
"""


class OcrError(RuntimeError):
    pass


@dataclass
class PageText:
    text: str
    language: str
    unreadable: bool = False


class PageReader(Protocol):
    async def read_page(self, image_path: Path) -> PageText: ...


class TextCleaner(Protocol):
    async def clean(self, text: str) -> PageText: ...


def _parse_payload(content: str | None) -> dict:
    if not content:
        raise OcrError("empty response from model")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OcrError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise OcrError("model response missing text field")
    return payload


class OpenAIVisionReader:
    def __init__(self, client: AsyncOpenAI, model: str, concurrency: int = 3) -> None:
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(concurrency)

    async def read_page(self, image_path: Path) -> PageText:
        mime = MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/jpeg")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"},
                        }
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        payload = _parse_payload(completion.choices[0].message.content)
        return PageText(
            text=payload["text"].strip(),
            language=str(payload.get("language") or "english").strip().lower(),
            unreadable=bool(payload.get("unreadable", False)),
        )


class OpenAITextCleaner:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def _clean_chunk(self, text: str) -> tuple[str, str]:
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        payload = _parse_payload(completion.choices[0].message.content)
        return payload["text"].strip(), str(payload.get("language") or "english").strip().lower()

    async def clean(self, text: str) -> PageText:
        chunks = chunk_text(text, max_chars=6000) if text.strip() else [""]
        results = await asyncio.gather(*(self._clean_chunk(chunk) for chunk in chunks))
        joined = "\n\n".join(cleaned for cleaned, _ in results if cleaned)
        language = results[0][1] if results else "english"
        return PageText(text=joined, language=language)
