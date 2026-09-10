# Telegram Reader

A private Telegram bot for reading printed pages by ear. Photograph pages of a book or
printed article, send them to the bot, tap **Read aloud** — and get an MP3 plus the
transcribed text. Built so someone recovering from a concussion can keep up with
thesis reading while resting from screens.

> **Note:** this project was mostly built by AI, with a human steering the requirements
> and reviewing the results.

```
You (phone)                Home server                     OpenAI
  📷 page photo ──────────▶ OCR (gpt-4.1-mini vision)  ───▶ clean text
  📄 PDF        ──────────▶ text layer / OCR per page ──▶ clean text
  🔊 Read aloud ──────────▶ TTS (gpt-4o-mini-tts)     ───▶ MP3 + transcript
```

## Usage

1. Send one or more photos of pages to the bot (an album works in one go). The bot reacts
   with 👀 and quietly OCRs each page in the background — nothing to read.
2. One message shows `📖 N/M pages ready` with a **🔊 Read aloud** button. Tap it (or send
   `/done`) when all pages are in.
3. The bot replies with the MP3 (titled after the document) and the text. Batches never
   finalize automatically — only on the button or `/done`.
4. The Telegram audio player keeps playing with the screen off, supports 1.5x/2x speed,
   and files stay in the chat's shared-media "Music" tab.

Commands: `/done` `/status` `/text` (resend last transcript) `/cancel` `/start` (help).

Tips: send an **album** for many pages at once. For small print, send the image as a
**file** (not a photo) to keep full resolution — Telegram compresses photos to 1280px.
PDFs are supported too (text layer used when present, scanned pages are OCR'd); Telegram
caps bot downloads at 20 MB, so split bigger PDFs.

## Setup

Requirements: Python 3.12+ with [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, and a
machine that is always on (home server).

1. **Create the bot**: in Telegram, talk to [@BotFather](https://t.me/BotFather) →
   `/newbot` → copy the token.
2. **Find your user ID**: message [@userinfobot](https://t.me/userinfobot) and copy the
   numeric id.
3. **Configure**:
   ```bash
   cp .env.example .env
   # edit .env: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, ALLOWED_USER_IDS
   ```
   Only the Telegram user IDs in `ALLOWED_USER_IDS` can use the bot; everyone else is
   refused.
4. **Run**:
   ```bash
   uv sync
   uv run python -m reader.bot
   ```
   The bot uses long polling — no open ports, no TLS, no tunnel needed.

### Docker

```bash
docker compose up -d --build
```

`docker compose logs -f` to watch it. Files are stored in `./data/` and purged after
`RETENTION_DAYS` (default 14).

### Run as a systemd user service

For the always-on setup (crash restart + boot autostart), the unit lives at
`~/.config/systemd/user/telegram-reader.service`:

```ini
[Unit]
Description=Telegram Reader bot

[Service]
WorkingDirectory=%h/Repos/telegram-reader
ExecStart=%h/Repos/telegram-reader/.venv/bin/python -m reader.bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Manage it with:

```bash
systemctl --user status telegram-reader      # is it running
journalctl --user -u telegram-reader -f      # follow the logs
systemctl --user restart telegram-reader     # after code updates
systemctl --user stop telegram-reader        # stop it
```

Boot autostart without an active login session requires linger (already enabled here):
`loginctl enable-linger $USER`. Update ritual: pull changes, `uv sync`, then restart the
service.

### Local testing without Telegram

Useful for checking OCR quality and how the Danish voice sounds before relying on it:

```bash
export OPENAI_API_KEY=sk-...
uv run python -m reader.cli page1.jpg page2.jpg -o out/
uv run python -m reader.cli article.pdf -o out/ --voice onyx
```

This prints the transcript path and MP3 parts without needing Telegram.

## Cost

Approximate, at current OpenAI prices: OCR ≈ $0.002/page (vision, detail=high), TTS
≈ $0.05/page → **about $0.05 per photographed page**. Reading a few hundred pages costs
roughly $10–25. PDFs with a text layer skip the vision call and cost only TTS + a small
cleanup pass.

## Privacy

- Page photos are sent to the OpenAI API for OCR; per their API policy, data is not used
  for training by default.
- Transcripts and audio stay on the server under `data/<user>/<session>/` and are deleted
  after `RETENTION_DAYS`.
- `/cancel` deletes the pending pages immediately.
- Access is restricted to `ALLOWED_USER_IDS`; everyone else is refused.

## Design notes & caveats

- **Danish TTS**: OpenAI's voices handle Danish intelligibly but with a non-native
  accent. If that gets tiring, the TTS layer is a small interface
  (`src/reader/tts.py`) — swapping in Azure Neural TTS (native `da-DK` voices, 500k
  characters/month free tier) is a contained change.
- OCR uses a vision model rather than Tesseract because it handles curved pages,
  two-column layouts, hyphenation across lines, and drops headers/footers/page numbers.
  It never translates or summarizes; the transcript is verbatim.
- Audio is assembled from ≤4096-char chunks (paragraph/sentence boundaries) via ffmpeg,
  split into parts sized for Telegram's 50 MB limit.
- If a page comes back unreadable, retake that photo — flatter, closer, more light.

## Development

```bash
uv sync                 # install deps
uv run pytest           # tests (31)
uv run ruff check src tests
```

Layout: `config.py` (env settings) · `textproc.py` (merge/chunk/title, pure) ·
`ocr.py` (vision OCR + PDF-text cleanup) · `tts.py` (TTS + ffmpeg concat) ·
`pdf.py` (PyMuPDF extraction/rendering) · `sessions.py` (per-user state on disk) ·
`pipeline.py` (document → Result) · `bot.py` (Telegram handlers) · `cli.py`.