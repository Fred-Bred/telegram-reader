import re
from collections.abc import Iterable

_LETTER = r"[^\W\d_]"
_HYPHEN_LINE_BREAK = re.compile(rf"({_LETTER}+)-\n({_LETTER}+)")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")

_KEEP_HYPHEN_PREFIXES = {
    "well",
    "long",
    "short",
    "self",
    "anti",
    "multi",
    "non",
    "co",
    "pre",
    "post",
    "semi",
    "vice",
    "full",
    "half",
    "socio",
    "ex",
}


def _should_keep_hyphen(left: str) -> bool:
    return left.lower() in _KEEP_HYPHEN_PREFIXES


def fix_hyphenation(text: str) -> str:
    def join(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        return _join_fragments(left, right)

    return _HYPHEN_LINE_BREAK.sub(join, text)


def _join_fragments(left: str, right: str) -> str:
    if right.islower() and not _should_keep_hyphen(left):
        return left + right
    return left + "-" + right


def merge_pages(pages: Iterable[str]) -> str:
    parts = [page.strip() for page in pages if page and page.strip()]
    if not parts:
        return ""
    merged = parts[0]
    for next_part in parts[1:]:
        if merged.endswith("-") and next_part[:1].islower():
            head = merged[:-1]
            left = head.split()[-1] if head.split() else ""
            if left:
                merged = head[: -len(left)] + _join_fragments(left, next_part)
            else:
                merged = head + next_part
        else:
            merged = merged + "\n\n" + next_part
    return merged


def guess_title(text: str) -> str | None:
    for line in text.splitlines()[:8]:
        candidate = line.strip()
        if 3 <= len(candidate) <= 120 and not candidate.endswith((".", ";", ",")):
            return candidate
    return None


def sanitize_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\'\x00-\x1f]+', " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".,")
    cleaned = re.sub(r" ([.,])", r"\1", cleaned)
    cleaned = cleaned[:max_len].strip()
    return cleaned or "document"


def _split_long_block(block: str, max_chars: int) -> list[str]:
    sentences = _SENTENCE_BOUNDARY.split(block)
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        candidate = f"{buf} {sentence}" if buf else sentence
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = sentence
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for paragraph in re.split(r"\n{2,}", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_long_block(paragraph, max_chars))
            continue
        candidate = f"{buf}\n\n{paragraph}" if buf else paragraph
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            chunks.append(buf)
            buf = paragraph
    if buf:
        chunks.append(buf)
    return chunks


def group_chunks(chunks: Iterable[str], part_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk) > part_chars:
            groups.append(current)
            current = []
            size = 0
        current.append(chunk)
        size += len(chunk)
    if current:
        groups.append(current)
    return groups
