import asyncio
import contextlib
import logging
from pathlib import Path

from openai import AsyncOpenAI
from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Message,
    Update,
)
from telegram.constants import ReactionEmoji
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings, get_settings
from .ocr import OpenAITextCleaner, OpenAIVisionReader
from .pdf import load_pdf_pages
from .pipeline import EmptyDocument, finalize_session, wait_for_pages
from .sessions import Page, Result, Session, SessionStore
from .tts import OpenAiTts, require_ffmpeg

logger = logging.getLogger(__name__)

HELP = """\
Send photos of pages — or a PDF — and tap 🔊 Read aloud when you have sent them all. \
You will get an audio recording and the text.

/done — generate audio for the pages sent so far
/status — how many pages are waiting
/text — resend the text of the last document
/cancel — discard the pending pages

Tips: send an album to add many pages at once. For small print, send the image as a \
file instead of a photo to keep full resolution. PDFs larger than 20 MB must be split.
"""

IMAGE_EXT_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}

READ_BUTTON = "read"
READ_BUTTON_LABEL = "🔊 Read aloud"


class ReaderBot:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        reader: OpenAIVisionReader,
        cleaner: OpenAITextCleaner,
        tts: OpenAiTts,
    ) -> None:
        self.settings = settings
        self.store = store
        self.reader = reader
        self.cleaner = cleaner
        self.tts = tts
        self.bot: Bot | None = None
        self._purge_task: asyncio.Task | None = None

    def build(self) -> Application:
        application = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .post_init(self.post_init)
            .build()
        )
        self.bot = application.bot
        application.add_handler(
            MessageHandler(
                filters.PHOTO | filters.Document.IMAGE | filters.Document.PDF, self.on_media
            )
        )
        application.add_handler(CommandHandler("start", self.cmd_start))
        application.add_handler(CommandHandler("done", self.cmd_done))
        application.add_handler(CommandHandler("cancel", self.cmd_cancel))
        application.add_handler(CommandHandler("status", self.cmd_status))
        application.add_handler(CommandHandler("text", self.cmd_text))
        application.add_handler(
            CallbackQueryHandler(self.on_read_button, pattern=f"^{READ_BUTTON}$")
        )
        return application

    async def post_init(self, application: Application) -> None:
        require_ffmpeg()
        await application.bot.set_my_commands(
            [
                BotCommand("done", "Generate audio for pending pages"),
                BotCommand("status", "Show pending page count"),
                BotCommand("text", "Resend the last transcript"),
                BotCommand("cancel", "Discard pending pages"),
            ]
        )
        self.resume_pending()
        self._purge_task = asyncio.create_task(self._purge_loop())

    def resume_pending(self) -> None:
        resumed = 0
        for user_id in self.store.all_user_ids():
            for session in self.store.sessions_for(user_id):
                if session.state != "collecting" or not session.pages:
                    continue
                for page in session.pages:
                    if page.status in {"pending", "working"}:
                        page.status = "pending"
                        self._dispatch(session, page)
                self.store.save(session)
                asyncio.create_task(self._refresh_button(session, session.button_chat_id))
                resumed += 1
        if resumed:
            logger.info("resumed %s session(s)", resumed)

    async def _purge_loop(self) -> None:
        while True:
            try:
                removed = self.store.purge_expired(self.settings.retention_days)
                if removed:
                    logger.info("purged %s expired sessions", len(removed))
            except Exception:
                logger.exception("purge failed")
            await asyncio.sleep(3600)

    async def _authorize(self, update: Update) -> bool:
        user = update.effective_user
        if user and user.id in self.settings.allowed_user_ids:
            return True
        logger.warning("rejected message from user %s", user.id if user else "unknown")
        message = update.effective_message
        if message:
            with contextlib.suppress(TelegramError):
                await message.reply_text("This bot is private.")
        return False

    async def on_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or not await self._authorize(update):
            return
        session = self.store.active(update.effective_user.id)
        if session is None or session.state != "collecting":
            session = self.store.create(update.effective_user.id)
        try:
            await message.set_reaction(ReactionEmoji.EYES)
            if message.photo:
                await self._add_image_page(session, message.photo[-1], ".jpg")
            elif message.document and (message.document.mime_type or "").startswith("image/"):
                mime = message.document.mime_type or "image/jpeg"
                extension = IMAGE_EXT_BY_MIME.get(mime, ".jpg")
                await self._add_image_page(session, message.document, extension)
            elif message.document and message.document.mime_type == "application/pdf":
                await self._add_pdf(session, message.document)
            else:
                return
        except TelegramError:
            logger.exception("download failed")
            await self._reply(message, "Could not download that file. PDFs must be under 20 MB.")
            return
        await self._refresh_button(session, message.chat_id)

    async def _add_image_page(self, session: Session, attachment, extension: str) -> None:
        index = len(session.pages) + 1
        path = session.dir / f"page_{index:03d}{extension}"
        tg_file = await attachment.get_file()
        await tg_file.download_to_drive(path)
        page = Page(index=index, kind="image", source=str(path))
        session.pages.append(page)
        self._dispatch(session, page)
        self.store.save(session)

    async def _add_pdf(self, session: Session, document) -> None:
        index = len(session.pages) + 1
        pdf_path = session.dir / f"upload_{index:03d}.pdf"
        tg_file = await document.get_file()
        await tg_file.download_to_drive(pdf_path)
        pdf_pages = await asyncio.to_thread(load_pdf_pages, pdf_path, session.dir)
        for pdf_page in pdf_pages:
            page_index = len(session.pages) + 1
            if pdf_page.text is not None:
                page = Page(index=page_index, kind="pdf_text", source=pdf_page.text)
            else:
                page = Page(index=page_index, kind="image", source=str(pdf_page.image_path))
            session.pages.append(page)
            self._dispatch(session, page)
        pdf_path.unlink(missing_ok=True)
        self.store.save(session)

    def _dispatch(self, session: Session, page: Page) -> None:
        task = asyncio.create_task(self._process_page(session, page))
        self.store.add_task(session, task)

    async def _process_page(self, session: Session, page: Page) -> None:
        page.status = "working"
        try:
            if page.kind == "image":
                result = await self.reader.read_page(Path(page.source))
            elif page.kind == "pdf_text":
                result = await self.cleaner.clean(page.source)
            else:
                raise RuntimeError(f"unknown page kind: {page.kind}")
            if result.unreadable:
                page.status = "unreadable"
            else:
                page.status = "ready"
                page.text = result.text
                page.language = result.language
        except Exception as exc:
            page.status = "failed"
            page.error = str(exc)[:300]
            logger.warning("page %s failed: %s", page.index, exc)
        finally:
            self.store.save(session)
            await self._refresh_button(session, session.button_chat_id)

    async def _refresh_button(self, session: Session, chat_id: int | None) -> None:
        if session.state != "collecting" or not session.pages or chat_id is None:
            return
        counts = session.counts()
        total = len(session.pages)
        text = f"📖 {counts.get('ready', 0)}/{total} pages ready"
        problems = counts.get("unreadable", 0) + counts.get("failed", 0)
        if problems:
            text += f" · {problems} to retake"
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton(READ_BUTTON_LABEL, callback_data=READ_BUTTON)]]
        )
        if session.button_chat_id is not None and session.button_message_id is not None:
            with contextlib.suppress(BadRequest):
                await self.bot.edit_message_text(
                    text,
                    chat_id=session.button_chat_id,
                    message_id=session.button_message_id,
                    reply_markup=keyboard,
                )
        else:
            sent = await self.bot.send_message(chat_id, text, reply_markup=keyboard)
            session.button_chat_id = chat_id
            session.button_message_id = sent.message_id
            self.store.save(session)

    async def on_read_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not await self._authorize(update):
            await query.answer()
            return
        await query.answer()
        session = None
        if query.message is not None:
            session = self.store.by_button(query.message.chat_id, query.message.message_id)
        if session is not None:
            await self.finalize(session)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorize(update):
            return
        await self._reply(update.effective_message, HELP)

    async def cmd_done(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorize(update):
            return
        session = self.store.active(update.effective_user.id)
        if session is None or not session.pages:
            await self._reply(update.effective_message, "No pending pages. Send photos first.")
            return
        if session.state == "finalizing":
            await self._reply(update.effective_message, "Already generating audio — one moment.")
            return
        await self.finalize(session)

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorize(update):
            return
        session = self.store.active(update.effective_user.id)
        if session is None or not session.pages:
            await self._reply(update.effective_message, "Nothing to cancel.")
            return
        pending = len(session.pages)
        self.store.drop(session)
        await self._reply(update.effective_message, f"Cleared {pending} pending pages.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorize(update):
            return
        session = self.store.active(update.effective_user.id)
        if session is None or not session.pages:
            await self._reply(update.effective_message, "No pending pages.")
            return
        await self._reply(update.effective_message, session.summary())

    async def cmd_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._authorize(update):
            return
        result = self.store.latest(update.effective_user.id)
        if result is None:
            await self._reply(update.effective_message, "No documents yet.")
            return
        transcript = Path(result.transcript_path)
        document = InputFile(
            transcript.read_bytes() if transcript.is_file() else result.text.encode("utf-8"),
            filename=f"{result.title}.txt",
        )
        if update.effective_chat is not None:
            await context.bot.send_document(update.effective_chat.id, document=document)

    async def finalize(self, session: Session) -> None:
        if session.state == "finalizing":
            if self.bot is not None and session.button_chat_id is not None:
                await self.bot.send_message(
                    session.button_chat_id, "Already generating audio for these pages."
                )
            return
        if session.state != "collecting" or self.bot is None:
            return
        if not session.pages:
            if session.button_chat_id is not None:
                await self.bot.send_message(
                    session.button_chat_id, "Send pages first, then tap Read aloud."
                )
            return
        chat_id = session.button_chat_id
        if chat_id is None:
            return
        session.state = "finalizing"
        self.store.save(session)
        await self.bot.send_message(chat_id, "⏳ Reading your pages…")
        await wait_for_pages(session, self.store.tasks_for(session))
        try:
            result = await finalize_session(session, self.tts, self.settings)
        except EmptyDocument:
            session.state = "collecting"
            self.store.save(session)
            await self.bot.send_message(
                chat_id, "No readable pages yet — send photos of the pages first."
            )
            return
        except Exception:
            logger.exception("finalize failed for session %s", session.id)
            session.state = "collecting"
            self.store.save(session)
            await self.bot.send_message(
                chat_id, "Something went wrong generating the audio. Try again with /done."
            )
            return
        session.result = result
        session.state = "done"
        self.store.save(session)
        self.store.set_latest(session)
        self.store.clear_tasks(session)
        for i, audio_path in enumerate(result.audio_paths, 1):
            part_title = result.title
            if len(result.audio_paths) > 1:
                part_title = f"{result.title} (part {i}/{len(result.audio_paths)})"
            audio = InputFile(Path(audio_path).read_bytes(), filename=Path(audio_path).name)
            await self.bot.send_audio(
                chat_id, audio=audio, title=part_title[:64], performer="Telegram Reader"
            )
        await self._deliver_text(session, chat_id, result)
        counts = session.counts()
        problems = counts.get("unreadable", 0) + counts.get("failed", 0)
        if problems:
            await self.bot.send_message(
                chat_id,
                f"{problems} page(s) could not be read — retake those photos and send them again.",
            )
        if session.button_message_id is not None:
            with contextlib.suppress(TelegramError):
                await self.bot.set_message_reaction(
                    chat_id, session.button_message_id, ReactionEmoji.THUMBS_UP
                )

    async def _deliver_text(self, session: Session, chat_id: int, result: Result) -> None:
        if len(result.text) <= self.settings.inline_text_max_chars:
            await self.bot.send_message(chat_id, result.text)
            return
        document = InputFile(result.text.encode("utf-8"), filename=f"{result.title}.txt")
        caption = f"{result.title} — transcript"
        await self.bot.send_document(chat_id, document=document, caption=caption)

    async def _reply(self, message: Message | None, text: str) -> None:
        if message is not None:
            await message.reply_text(text)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    require_ffmpeg()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    store = SessionStore(settings.data_dir)
    store.load()
    reader = OpenAIVisionReader(client, settings.ocr_model, settings.ocr_concurrency)
    cleaner = OpenAITextCleaner(client, settings.cleanup_model)
    tts = OpenAiTts(client, settings.tts_model, settings.tts_voice)
    bot = ReaderBot(settings, store, reader, cleaner, tts)
    application = bot.build()
    logger.info("starting bot for users: %s", settings.allowed_user_ids)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
