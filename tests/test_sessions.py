

from datetime import UTC, datetime

import reader.sessions as sessions_module  # noqa: F401
from reader.sessions import Page, Result, Session, SessionStore


def make_session(user_id=42, **kwargs) -> Session:
    defaults = dict(
        id=f"{user_id}-abc",
        user_id=user_id,
        dir=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    defaults.update(kwargs)
    return Session(**defaults)


def test_store_create_and_load(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(42)
    session.pages.append(Page(index=1, kind="image", source=str(tmp_path / "p1.jpg")))
    session.state = "done"
    session.result = Result(
        title="T", language="danish", text="hello", audio_paths=["a.mp3"], transcript_path="t.txt"
    )
    store.save(session)

    store2 = SessionStore(tmp_path)
    store2.load()
    loaded = store2.sessions_for(42)[0]
    assert loaded.state == "done"
    assert loaded.pages[0].kind == "image"
    assert loaded.result.title == "T"


def test_store_active_returns_last_collecting(tmp_path):
    store = SessionStore(tmp_path)
    assert store.active(1) is None
    first = store.create(1)
    first.state = "done"
    second = store.create(1)
    assert store.active(1) is second


def test_store_by_button(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(7)
    session.button_chat_id = 7
    session.button_message_id = 99
    assert store.by_button(7, 99) is session
    assert store.by_button(7, 100) is None


def test_store_latest_roundtrip(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(5)
    session.result = Result(
        title="My Title",
        language="english",
        text="abc",
        audio_paths=["/x/y.mp3"],
        transcript_path="/x/t.txt",
    )
    store.set_latest(session)
    latest = store.latest(5)
    assert latest is not None and latest.title == "My Title"
    assert store.latest(6) is None


def test_store_drop_removes_dir(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(3)
    assert session.dir.is_dir()
    store.drop(session)
    assert not session.dir.exists()
    assert store.active(3) is None


def test_store_purge_expired(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(9)
    session.created_at = "2020-01-01T00:00:00+00:00"
    removed = store.purge_expired(retention_days=14)
    assert removed == [session]
    assert store.sessions_for(9) == []


def test_page_counts_and_summary():
    session = make_session(user_id=42, dir=None)
    session.pages = [
        Page(index=1, kind="image", source="a", status="ready"),
        Page(index=2, kind="image", source="b", status="working"),
        Page(index=2, kind="image", source="b", status="unreadable"),
    ]
    assert session.summary() == "3 pages: 1 ready, 1 processing, 1 unreadable"
