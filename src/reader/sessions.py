import asyncio
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

COLLECTING_STATES = {"collecting", "finalizing"}


@dataclass
class Page:
    index: int
    kind: str
    source: str
    status: str = "pending"
    text: str | None = None
    language: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "text": self.text,
            "language": self.language,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Page":
        return cls(
            index=int(data["index"]),
            kind=str(data["kind"]),
            source=str(data["source"]),
            status=str(data.get("status", "pending")),
            text=data.get("text"),
            language=data.get("language"),
            error=data.get("error"),
        )


@dataclass
class Result:
    title: str
    language: str
    text: str
    audio_paths: list[str]
    transcript_path: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "language": self.language,
            "text": self.text,
            "audio_paths": self.audio_paths,
            "transcript_path": self.transcript_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        return cls(
            title=str(data["title"]),
            language=str(data.get("language", "english")),
            text=str(data.get("text", "")),
            audio_paths=[str(p) for p in data.get("audio_paths", [])],
            transcript_path=str(data.get("transcript_path", "")),
        )


@dataclass
class Session:
    id: str
    user_id: int
    dir: Path
    created_at: str
    state: str = "collecting"
    pages: list[Page] = field(default_factory=list)
    button_chat_id: int | None = None
    button_message_id: int | None = None
    result: Result | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "dir": str(self.dir),
            "created_at": self.created_at,
            "state": self.state,
            "pages": [page.to_dict() for page in self.pages],
            "button_chat_id": self.button_chat_id,
            "button_message_id": self.button_message_id,
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        result_data = data.get("result")
        return cls(
            id=str(data["id"]),
            user_id=int(data["user_id"]),
            dir=Path(str(data["dir"])),
            created_at=str(data["created_at"]),
            state=str(data.get("state", "collecting")),
            pages=[Page.from_dict(page) for page in data.get("pages", [])],
            button_chat_id=data.get("button_chat_id"),
            button_message_id=data.get("button_message_id"),
            result=Result.from_dict(result_data) if result_data else None,
        )

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page in self.pages:
            counts[page.status] = counts.get(page.status, 0) + 1
        return counts

    def summary(self) -> str:
        counts = self.counts()
        pieces = [f"{counts.get('ready', 0)} ready"]
        pending = counts.get("pending", 0) + counts.get("working", 0)
        if pending:
            pieces.append(f"{pending} processing")
        if counts.get("unreadable"):
            pieces.append(f"{counts['unreadable']} unreadable")
        if counts.get("failed"):
            pieces.append(f"{counts['failed']} failed")
        return f"{len(self.pages)} pages: " + ", ".join(pieces)

    @classmethod
    def new(cls, user_id: int, data_dir: Path) -> "Session":
        stamp = uuid.uuid4().hex[:8]
        directory = data_dir / str(user_id) / stamp
        return cls(
            id=f"{user_id}-{stamp}",
            user_id=user_id,
            dir=directory,
            created_at=datetime.now(UTC).isoformat(),
        )


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[int, list[Session]] = {}
        self._tasks: dict[str, list[asyncio.Task]] = {}

    def load(self) -> None:
        for user_dir in sorted(self.data_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            for session_dir in sorted(user_dir.iterdir()):
                session_file = session_dir / "session.json"
                if not session_file.is_file():
                    continue
                try:
                    session = Session.from_dict(
                        json.loads(session_file.read_text(encoding="utf-8"))
                    )
                except Exception:
                    continue
                self._sessions.setdefault(session.user_id, []).append(session)

    def all_user_ids(self) -> list[int]:
        return sorted(
            {session.user_id for sessions in self._sessions.values() for session in sessions}
        )

    def sessions_for(self, user_id: int) -> list[Session]:
        return self._sessions.get(user_id, [])

    def active(self, user_id: int) -> Session | None:
        for session in reversed(self.sessions_for(user_id)):
            if session.state in COLLECTING_STATES:
                return session
        return None

    def create(self, user_id: int) -> Session:
        session = Session.new(user_id, self.data_dir)
        self._sessions.setdefault(user_id, []).append(session)
        session.dir.mkdir(parents=True, exist_ok=True)
        self.save(session)
        return session

    def by_button(self, chat_id: int, message_id: int) -> Session | None:
        for sessions in self._sessions.values():
            for session in sessions:
                if session.button_chat_id == chat_id and session.button_message_id == message_id:
                    return session
        return None

    def add_task(self, session: Session, task: asyncio.Task) -> None:
        self._tasks.setdefault(session.id, []).append(task)

    def tasks_for(self, session: Session) -> list[asyncio.Task]:
        return self._tasks.get(session.id, [])

    def clear_tasks(self, session: Session) -> list[asyncio.Task]:
        return self._tasks.pop(session.id, [])

    def save(self, session: Session) -> None:
        session.dir.mkdir(parents=True, exist_ok=True)
        target = session.dir / "session.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(tmp, target)

    def set_latest(self, session: Session) -> None:
        if session.result is None:
            return
        latest = {
            "session_id": session.id,
            "dir": str(session.dir),
            "created_at": session.created_at,
            "result": session.result.to_dict(),
        }
        path = self.data_dir / str(session.user_id) / "latest.json"
        path.write_text(json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")

    def latest(self, user_id: int) -> Result | None:
        path = self.data_dir / str(user_id) / "latest.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Result.from_dict(data["result"])
        except Exception:
            return None

    def drop(self, session: Session) -> None:
        sessions = self._sessions.get(session.user_id, [])
        if session in sessions:
            sessions.remove(session)
        self._tasks.pop(session.id, None)
        shutil.rmtree(session.dir, ignore_errors=True)

    def purge_expired(self, retention_days: int) -> list[Session]:
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        removed: list[Session] = []
        for user_id in list(self._sessions):
            for session in list(self._sessions[user_id]):
                try:
                    created = datetime.fromisoformat(session.created_at)
                except ValueError:
                    continue
                if created < cutoff:
                    self._sessions[user_id].remove(session)
                    self._tasks.pop(session.id, None)
                    shutil.rmtree(session.dir, ignore_errors=True)
                    removed.append(session)
        return removed
