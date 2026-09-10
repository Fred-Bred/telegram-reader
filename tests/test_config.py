from reader.config import Settings


def make(monkeypatch, ids):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:t")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    if ids is not None:
        monkeypatch.setenv("ALLOWED_USER_IDS", ids)
    return Settings(_env_file=None)


def test_ids_csv(monkeypatch):
    assert make(monkeypatch, "11,22,33").allowed_user_ids == [11, 22, 33]


def test_ids_json_list(monkeypatch):
    assert make(monkeypatch, "[11, 22]").allowed_user_ids == [11, 22]


def test_ids_empty(monkeypatch):
    assert make(monkeypatch, "").allowed_user_ids == []


def test_defaults(monkeypatch):
    settings = make(monkeypatch, "1")
    assert settings.tts_voice == "sage"
    assert settings.tts_chunk_chars == 4000
    assert settings.tts_part_chars == 25000
    assert settings.retention_days == 14
    assert settings.inline_text_max_chars == 1500
