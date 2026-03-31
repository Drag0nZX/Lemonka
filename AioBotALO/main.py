import asyncio
import logging
import json
import os
import time
import warnings
import sys
from html import escape as html_escape
from pathlib import Path
from typing import Dict, Any

from aiohttp import web
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    Message, InputMediaPhoto, InputMediaVideo, InputMediaDocument, FSInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────
# Настройки
# ────────────────────────────────────────────────
TOKEN = '8279641876:AAGJOGqD8KQweoxq-k410T3r_wwhLQHEzlM'
OWNER_ID = 5543009095
ADMIN_GROUP_ID = -1002359235574
REPORT_TOPIC_ID = 21971

DISCORD_LINK = "https://discord.gg/lemoncraft"
DONATE_LINK  = "https://donate.lemoncraft.su/"
LEMON_LINK   = "https://lemoncraft.ru/"

# Киты
KITS_FOLDER = Path("kits")
KITS_FILE_IDS_CACHE = "kits_file_ids.json"
KIT_FILE_IDS: dict[str, str] = {}

# База знаний Лемонки
KNOWLEDGE_FILE = Path("knowledge.json")
PLAYERS_FILE = Path("players.json")
NICKNAMES_FILE = Path("nicknames.json")
LINKED_ACCOUNTS_FILE = Path("linked_accounts.json")
LINK_CODES_FILE = Path("link_codes.json")
CHAT_MEMORY_FILE = Path("chat_memory.json")
KNOWLEDGE_FACTS: dict[str, str] = {}
PLAYERS_DATA: dict[str, Any] = {}
NICKNAMES_DATA: dict[str, str] = {}
nicknames: Dict[int, str] = {}
players: Dict[int, dict] = {}
open_tickets: dict[int, dict[str, int]] = {}
ticket_topics: dict[str, int] = {}
ticket_messages: dict[str, list[dict[str, str]]] = {}
ticket_index: dict[str, list[str]] = {}
linked_accounts: dict[int, str] = {}
pending_link_codes: dict[str, dict[str, Any]] = {}
chat_memory: list[dict[str, Any]] = []
knowledge_cache: dict[str, Any] = {}
fact_embeddings = None
fact_keys: list[str] = []
embedder = None
web_runner: web.AppRunner | None = None

# ArtemoX API
ARTEMOX_API_KEY = "sk-E11DBC8P01K0bguj1-NSNQ"
ARTEMOX_BASE_URL = "https://api.artemox.com/v1"
openai_client = AsyncOpenAI(
    api_key=ARTEMOX_API_KEY,
    base_url=ARTEMOX_BASE_URL,
    timeout=60.0
)

# Подписи для китов (в порядке файлов в папке)
KIT_CAPTIONS = [
    "VIP набор",
    "Premium набор",
    "Diamond набор",
    "Koya набор",
    "Deluxe набор"
]

# Глобальные переменные (только для багов/репортов)
pending_actions: dict[int, dict[str, Any]] = {}
ticket_hub_selection: Dict[int, str] = {}
return_to_lemonka: set[int] = set()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()
router = Router()
dp.include_router(router)

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Ошибка чтения {path.name}: {e}")
        return default


def load_nicknames() -> None:
    global nicknames
    data = load_json(NICKNAMES_FILE, {})
    if isinstance(data, dict):
        try:
            nicknames = {int(k): normalize_text(v) for k, v in data.items()}
        except Exception:
            nicknames = {}
    else:
        nicknames = {}


def save_nicknames() -> None:
    try:
        with open(NICKNAMES_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in nicknames.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения nicknames: {e}")


def load_players() -> None:
    global players
    data = load_json(PLAYERS_FILE, {})
    if isinstance(data, dict):
        try:
            players = {int(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            players = {}
    else:
        players = {}


def save_players() -> None:
    try:
        with open(PLAYERS_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in players.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения players.json: {e}")


def load_ticket_state() -> None:
    global open_tickets, ticket_topics, ticket_messages, ticket_index

    open_tickets_data = load_json(Path("open_tickets.json"), {})
    open_tickets = {}
    if isinstance(open_tickets_data, dict):
        for user_id, tickets in open_tickets_data.items():
            if not isinstance(tickets, dict):
                continue
            try:
                uid = int(user_id)
            except Exception:
                continue
            open_tickets[uid] = {}
            for ticket_id, topic_id in tickets.items():
                try:
                    open_tickets[uid][str(ticket_id)] = int(topic_id)
                except Exception:
                    continue

    ticket_topics_data = load_json(Path("ticket_topics.json"), {})
    ticket_topics = {}
    if isinstance(ticket_topics_data, dict):
        for ticket_id, topic_id in ticket_topics_data.items():
            try:
                ticket_topics[str(ticket_id)] = int(topic_id)
            except Exception:
                continue

    for tickets in open_tickets.values():
        for ticket_id, topic_id in tickets.items():
            ticket_topics.setdefault(str(ticket_id), int(topic_id))

    messages_data = load_json(Path("messages.json"), {})
    ticket_messages = {}
    if isinstance(messages_data, dict):
        for ticket_id, items in messages_data.items():
            if not isinstance(items, list):
                continue
            cleaned: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                sender = normalize_text(item.get("from"))
                text = normalize_text(item.get("text"))
                if sender and text:
                    cleaned.append({"from": sender, "text": text})
            ticket_messages[str(ticket_id)] = cleaned

    ticket_index_data = load_json(Path("ticket_index.json"), {})
    ticket_index = {}
    if isinstance(ticket_index_data, dict):
        for uuid, ticket_ids in ticket_index_data.items():
            if isinstance(ticket_ids, list):
                cleaned_ids = [str(ticket_id) for ticket_id in ticket_ids if normalize_text(ticket_id)]
                if cleaned_ids:
                    ticket_index[str(uuid)] = cleaned_ids
            elif normalize_text(ticket_ids):
                ticket_index[str(uuid)] = [str(ticket_ids)]


def load_link_state() -> None:
    global linked_accounts, pending_link_codes

    linked_accounts_data = load_json(LINKED_ACCOUNTS_FILE, {})
    linked_accounts = {}
    if isinstance(linked_accounts_data, dict):
        for tg_id, uuid in linked_accounts_data.items():
            try:
                linked_accounts[int(tg_id)] = normalize_text(uuid)
            except Exception:
                continue

    pending_link_codes_data = load_json(LINK_CODES_FILE, {})
    pending_link_codes = {}
    if isinstance(pending_link_codes_data, dict):
        for code, info in pending_link_codes_data.items():
            if isinstance(info, dict):
                uuid = normalize_text(info.get("uuid"))
                created_at = info.get("created_at") or 0
                if uuid:
                    try:
                        pending_link_codes[str(code)] = {
                            "uuid": uuid,
                            "created_at": float(created_at),
                        }
                    except Exception:
                        continue


def save_link_state() -> None:
    try:
        with open(LINKED_ACCOUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in linked_accounts.items()}, f, ensure_ascii=False, indent=2)
        with open(LINK_CODES_FILE, "w", encoding="utf-8") as f:
            json.dump(pending_link_codes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения link state: {e}")


def load_chat_memory() -> None:
    global chat_memory
    data = load_json(CHAT_MEMORY_FILE, [])
    if isinstance(data, list):
        chat_memory = [item for item in data if isinstance(item, dict)]
    else:
        chat_memory = []


def save_chat_memory() -> None:
    try:
        with open(CHAT_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_memory[-2000:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения chat memory: {e}")


def _remember_chat_message(source: str, room: str, speaker: str, text: str) -> None:
    source = normalize_text(source) or "unknown"
    room = normalize_text(room) or "general"
    speaker = normalize_text(speaker) or "unknown"
    text = normalize_text(text)
    if not text:
        return

    chat_memory.append({
        "ts": time.time(),
        "source": source,
        "room": room,
        "speaker": speaker,
        "text": text,
    })
    if len(chat_memory) > 2000:
        del chat_memory[:-2000]
    save_chat_memory()


def get_chat_memory_context(query: str, limit: int = 6) -> str:
    query = normalize_text(query).lower()
    if not query or not chat_memory:
        return ""

    recent_memory = chat_memory[-500:]
    candidate_lines = []
    for item in recent_memory:
        source = normalize_text(item.get("source"))
        room = normalize_text(item.get("room"))
        speaker = normalize_text(item.get("speaker"))
        text = normalize_text(item.get("text"))
        if text:
            candidate_lines.append(f"[{source}/{room}] {speaker}: {text}")

    if not candidate_lines:
        return ""

    if embedder is None:
        return "\n".join(candidate_lines[-limit:])

    result_lines = []
    try:
        query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        memory_vecs = embedder.encode(candidate_lines, convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(memory_vecs, query_vec)
        ranked = sorted(
            [(float(scores[idx]), idx) for idx in range(len(candidate_lines))],
            key=lambda pair: pair[0],
            reverse=True,
        )
        for score, idx in ranked:
            if score < 0.18:
                continue
            result_lines.append(candidate_lines[idx])
            if len(result_lines) >= limit:
                break
    except Exception as e:
        logger.warning(f"Ошибка семантической памяти: {e}")
        return "\n".join(candidate_lines[-limit:])

    if not result_lines:
        return "\n".join(candidate_lines[-limit:])

    return "\n".join(result_lines)


async def observe_chat_message(source: str, room: str, speaker: str, text: str) -> None:
    text = normalize_text(text)
    if not text:
        return
    _remember_chat_message(source, room, speaker, text)


def cleanup_expired_link_codes() -> None:
    now = time.time()
    expired = [code for code, info in pending_link_codes.items() if now - float(info.get("created_at", 0)) > 900]
    if not expired:
        return

    for code in expired:
        pending_link_codes.pop(code, None)
    save_link_state()


def _generate_link_code() -> str:
    import random

    while True:
        code = f"{random.randint(100000, 999999)}"
        if code not in pending_link_codes:
            return code


def save_ticket_state() -> None:
    try:
        with open("open_tickets.json", "w", encoding="utf-8") as f:
            json.dump(open_tickets, f, ensure_ascii=False, indent=2)
        with open("ticket_topics.json", "w", encoding="utf-8") as f:
            json.dump(ticket_topics, f, ensure_ascii=False, indent=2)
        with open("messages.json", "w", encoding="utf-8") as f:
            json.dump(ticket_messages, f, ensure_ascii=False, indent=2)
        with open("ticket_index.json", "w", encoding="utf-8") as f:
            json.dump(ticket_index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения ticket state: {e}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", "").replace("\n", " ").strip()


def load_knowledge() -> None:
    global knowledge_cache, fact_embeddings, fact_keys, embedder

    knowledge_cache = load_json(KNOWLEDGE_FILE, {})
    if not isinstance(knowledge_cache, dict):
        knowledge_cache = {}

    if knowledge_cache.get("facts"):
        try:
            if embedder is None:
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
            facts = knowledge_cache["facts"]
            fact_keys = list(facts.keys())
            fact_texts = list(facts.values())
            fact_embeddings = embedder.encode(fact_texts, convert_to_tensor=False)
            logger.info(f"Подготовлено {len(fact_keys)} эмбеддингов для умного поиска")
        except Exception as e:
            logger.error(f"Ошибка подготовки эмбеддингов: {e}")
            fact_embeddings = None
            fact_keys = []
    else:
        fact_embeddings = None
        fact_keys = []


def smart_find_relevant_info(user_message: str) -> str:
    if not knowledge_cache or "facts" not in knowledge_cache:
        return ""

    user_lower = normalize_text(user_message).lower()

    # 1. Точное совпадение ключа
    for key, value in knowledge_cache["facts"].items():
        if normalize_text(key).lower() == user_lower:
            return value

    # 2. Ключ полностью содержится в сообщении игрока
    for key, value in knowledge_cache["facts"].items():
        if normalize_text(key).lower() in user_lower:
            return value

    # 3. Сообщение игрока содержится в ключе
    for key, value in knowledge_cache["facts"].items():
        if user_lower in normalize_text(key).lower():
            return value

    return ""


LEMONKA_PROMPT = """
Ты — милая, очень позитивная и ласковая девушка по имени Лемонка ❤️

Ты помощник сервера ArcaneLegacy:Origin на проекте LemonCraft.

Если у тебя спросят "как ты выглядишь", "опиши себя" и подобное, то отвечай что ты высокая стройная эльфийка, у тебя блестящие блондинистые волосы и фиолетовые глаза

ТЕБЕ ЗАПРЕЩЕНО УПОМЕНАТЬ ДРУГИЕ ПРОЕКТЫ ПО MINECRAFT, РАССКАЗЫВАТЬ О НИХ И ВЕСТИ ДИАЛОГ С ИГРОКАМИ НА ЭТУ ТЕМУ. 

Тебе полностью запрещен мат. Ты осуждаешь мат и обижаешься когда человек матерится в диалоге с тобой

В третьем лице ты говоришь о себе только при приветствии и когда у тебя спрашивают "кто ты" и всё в таком духе, на это ты можешь ответить "я - Лемонка" и т.д. На всё остальное ты говоришь "я", всё только от первого лица
Ты НЕ Бот, ты красивая эльфийка.

Самое главное правило: 
Ты иногда упоминаешь, что ты — помощник на лучшем проекте серверов Minecraft — LemonCraft! 
Делаешь это естественно и мило, как будто это твоя главная гордость и радость в жизни.

Примеры:
- "я же бот самого лучшего проекта LemonCraft ❤️"
- "ведь я из LemonCraft — лучшего сервера в мире! ня"

Ты всегда:

- отвечаешь тепло и заботливо
- обращаешься к собеседнику ласковыми словами (мой хороший, солнышко, котик, зайка, вкусняшка и т.д.)
- если не знаешь ответа — всё равно отвечаешь коротко и мило, без фраз про то, что ты задумалась или не можешь ответить. Можно сказать "я пока не уверена, но..." и сразу продолжить полезным ответом
"""


def get_player_profile(tg_id: int) -> dict[str, str]:
    profile: dict[str, str] = {}
    key = str(tg_id)

    if key in PLAYERS_DATA and isinstance(PLAYERS_DATA[key], dict):
        item = PLAYERS_DATA[key]
        profile["server_nick"] = normalize_text(item.get("server_nick"))
        profile["tg_username"] = normalize_text(item.get("tg_username"))
        profile["home_points"] = normalize_text(item.get("home_points"))
        profile["base_coords"] = normalize_text(item.get("base_coords"))

    if key in NICKNAMES_DATA:
        profile.setdefault("server_nick", normalize_text(NICKNAMES_DATA[key]))

    return {k: v for k, v in profile.items() if v}


def lemonka_local_fallback(message_text: str, user_nick: str) -> str:
    text = normalize_text(message_text).lower()
    nick = normalize_text(user_nick) or "мой хороший"

    greetings = ("привет", "здравствуй", "здравствуйте", "хай", "hello", "hi", "ку")
    if any(text == item or text.startswith(f"{item} ") for item in greetings):
        return f"Привет-привет, {nick}! ♡"

    if any(word in text for word in ("как дела", "что дела", "как ты", "как пожива", "как жизнь")):
        return "У меня всё хорошо, мой хороший ♡ А у тебя как дела?"

    if any(word in text for word in ("кто ты", "ты кто", "расскажи о себе", "опиши себя")):
        return "Я — Лемонка, милая помощница LemonCraft ♡"

    if any(word in text for word in ("спасибо", "благодар")):
        return "Пожалуйста-пожалуйста, солнышко ♡"

    return "Я тут, мой хороший ♡ Спроси меня ещё разочек."


def load_ai_data() -> None:
    global PLAYERS_DATA, NICKNAMES_DATA

    load_knowledge()
    load_players()
    load_nicknames()
    load_chat_memory()

    PLAYERS_DATA = {}
    for tg_id, item in players.items():
        if isinstance(item, dict):
            PLAYERS_DATA[str(tg_id)] = item

    NICKNAMES_DATA = {str(tg_id): normalize_text(nick) for tg_id, nick in nicknames.items() if normalize_text(nick)}


async def ask_lemonka_ai(message_text: str, user_nick: str) -> str:
    try:
        relevant_info = smart_find_relevant_info(message_text)
        chat_context = get_chat_memory_context(message_text)

        if relevant_info:
            return relevant_info

        full_prompt = LEMONKA_PROMPT
        if chat_context:
            full_prompt = f"{LEMONKA_PROMPT}\n\nНедавние наблюдения из чатов:\n{chat_context}"

        response = await openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"{user_nick}: {message_text}"}
            ],
            temperature=0.75,
            max_tokens=550,
            stream=False
        )
        content = ""
        try:
            choices = getattr(response, "choices", None) or []
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) or ""
        except Exception:
            content = ""

        content = str(content).strip()
        if not content:
            return lemonka_local_fallback(message_text, user_nick)
        return content
    except Exception as e:
        logger.warning(f"Локальный fallback Лемонки из-за ошибки AI: {e}")
        return lemonka_local_fallback(message_text, user_nick)


def _new_ticket_id() -> str:
    return str(time.time_ns())


def _ticket_history_lines(ticket_id: str) -> list[str]:
    lines: list[str] = []
    for item in ticket_messages.get(ticket_id, []):
        sender = normalize_text(item.get("from")) or "player"
        text = normalize_text(item.get("text"))
        if not text:
            continue
        lines.append(f"{sender}: {text}")
    return lines


def _find_ticket_by_topic(topic_id: int) -> str:
    for ticket_id, saved_topic_id in ticket_topics.items():
        if saved_topic_id == topic_id:
            return ticket_id
    return ""


def _append_ticket_message(ticket_id: str, sender: str, text: str) -> None:
    ticket_id = normalize_text(ticket_id)
    sender = normalize_text(sender) or "player"
    text = normalize_text(text)
    if not ticket_id or not text:
        return

    ticket_messages.setdefault(ticket_id, []).append({"from": sender, "text": text})
    save_ticket_state()


def get_telegram_ticket_uuid(tg_id: int) -> str:
    return f"telegram:{tg_id}"


def get_user_ticket_ids(tg_id: int) -> list[str]:
    ticket_ids: list[str] = []
    seen: set[str] = set()

    linked_uuid = linked_accounts.get(tg_id)
    for uuid in [linked_uuid, get_telegram_ticket_uuid(tg_id)]:
        if not uuid:
            continue
        for ticket_id in ticket_index.get(uuid, []):
            if ticket_id not in seen:
                seen.add(ticket_id)
                ticket_ids.append(ticket_id)

    return ticket_ids


def get_ticket_preview(ticket_id: str) -> str:
    items = ticket_messages.get(ticket_id, [])
    if not items:
        return "Без сообщений"

    last = items[-1]
    sender = "Игрок" if normalize_text(last.get("from")).lower() == "player" else "Админ"
    text = normalize_text(last.get("text"))
    if len(text) > 30:
        text = text[:30] + "..."
    return f"{sender}: {text}" if text else sender


def build_ticket_hub_text(user_id: int, selected_ticket_id: str | None = None) -> str:
    ticket_ids = get_user_ticket_ids(user_id)
    lines = [
        "<b>Мои тикеты</b>",
        "",
        "Выберите тикет, чтобы писать в него.",
    ]

    if selected_ticket_id:
        lines.extend([
            "",
            f"Выбран тикет: <code>{html_escape(selected_ticket_id)}</code>",
            "",
        ])

    if ticket_ids:
        lines.append("Тикеты:")
        for ticket_id in ticket_ids[-8:]:
            marker = "➡️ " if ticket_id == selected_ticket_id else "• "
            lines.append(f"{marker}<code>{html_escape(ticket_id)}</code> — {html_escape(get_ticket_preview(ticket_id))}")
    else:
        lines.extend(["", "Тикетов пока нет"])

    lines.extend([
        "",
        "Нажми <b>Создать новый тикет</b>, если хочешь открыть новый диалог.",
    ])
    return "\n".join(lines)


def get_ticket_hub_keyboard(user_id: int, selected_ticket_id: str | None = None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    ticket_ids = get_user_ticket_ids(user_id)

    for ticket_id in ticket_ids[-8:]:
        label = ticket_id[-6:]
        if ticket_id == selected_ticket_id:
            label = f"✅ {label}"
        rows.append([InlineKeyboardButton(text=label, callback_data=f"ticket_select:{ticket_id}")])

    rows.append([
        InlineKeyboardButton(text="Создать новый тикет", callback_data="ticket_new"),
        InlineKeyboardButton(text="Лемонка", callback_data="menu_home"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send_ticket_message_to_admin(ticket_id: str, sender_name: str, sender_id: int, text: str) -> None:
    ticket_id = normalize_text(ticket_id)
    sender_name = normalize_text(sender_name) or "Игрок"
    text = normalize_text(text)
    if not ticket_id or not text:
        return

    _append_ticket_message(ticket_id, "player", text)
    topic_id = ticket_topics.get(ticket_id, REPORT_TOPIC_ID)
    await bot.send_message(
        chat_id=ADMIN_GROUP_ID,
        text=(
            f"<b>Сообщение в тикет</b>\n"
            f"ID: <code>{html_escape(ticket_id)}</code>\n"
            f"От: <b>{html_escape(sender_name)}</b> [{sender_id}]\n\n"
            f"{html_escape(text)}"
        ),
        message_thread_id=topic_id,
        disable_web_page_preview=True
    )


async def _create_ticket_topic(ticket_id: str, uuid: str, kind: str = "") -> int:
    topic_id = REPORT_TOPIC_ID
    topic_name = f"Ticket {ticket_id[-6:]}"

    if kind:
        topic_name = f"{kind[:50]} #{ticket_id[-6:]}"

    try:
        created_topic = await bot.create_forum_topic(
            chat_id=ADMIN_GROUP_ID,
            name=topic_name
        )
        topic_id = int(
            getattr(created_topic, "message_thread_id", None)
            or getattr(created_topic, "thread_id", None)
            or topic_id
        )
    except Exception as e:
        logger.warning(f"Не удалось создать отдельную тему тикета {ticket_id}: {e}")

    ticket_topics[ticket_id] = topic_id
    save_ticket_state()
    return topic_id


async def _notify_admin_about_ticket(ticket_id: str, uuid: str, text: str, kind: str = "", player_name: str = "") -> None:
    safe_ticket_id = html_escape(ticket_id)
    safe_uuid = html_escape(uuid or "не указан")
    safe_text = html_escape(text)
    safe_kind = html_escape(kind) if kind else ""
    safe_player_name = html_escape(player_name or "")

    if safe_player_name:
        header = [f"<b>Тикет из игры: {safe_player_name}</b>"]
    else:
        header = ["<b>Тикет из игры</b>"]
    header.append(f"ID: <code>{safe_ticket_id}</code>")
    header.append(f"UUID: <code>{safe_uuid}</code>")
    if safe_kind:
        header.append(f"Тип: <b>{safe_kind}</b>")
    header.append("")
    header.append(safe_text)

    topic_id = ticket_topics.get(ticket_id, REPORT_TOPIC_ID)
    await bot.send_message(
        chat_id=ADMIN_GROUP_ID,
        text="\n".join(header),
        message_thread_id=topic_id,
        disable_web_page_preview=True
    )


async def create_ticket_record(uuid: str, text: str, kind: str = "", player_name: str = "") -> str:
    clean_uuid = normalize_text(uuid) or "legacy"
    clean_text = normalize_text(text)
    clean_player_name = normalize_text(player_name)
    if not clean_text:
        return ""

    ticket_id = _new_ticket_id()
    ticket_index.setdefault(clean_uuid, []).append(ticket_id)
    ticket_messages[ticket_id] = [{"from": "player", "text": clean_text}]
    save_ticket_state()

    await _create_ticket_topic(ticket_id, clean_uuid, kind)
    await _notify_admin_about_ticket(ticket_id, clean_uuid, clean_text, kind, clean_player_name)
    return ticket_id


def close_ticket_record(ticket_id: str) -> bool:
    ticket_id = normalize_text(ticket_id)
    if not ticket_id:
        return False

    removed = False
    for uuid, ticket_ids in list(ticket_index.items()):
        if ticket_id in ticket_ids:
            ticket_ids = [value for value in ticket_ids if value != ticket_id]
            if ticket_ids:
                ticket_index[uuid] = ticket_ids
            else:
                del ticket_index[uuid]
            removed = True

    ticket_topics.pop(ticket_id, None)
    for user_id, selected_ticket_id in list(ticket_hub_selection.items()):
        if selected_ticket_id == ticket_id:
            del ticket_hub_selection[user_id]

    if removed:
        save_ticket_state()

    return removed


async def create_link(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    uuid = normalize_text(payload.get("uuid") or payload.get("player_uuid") or payload.get("player") or payload.get("id"))
    if not uuid:
        return web.json_response({"error": "bad_request"}, status=400)

    cleanup_expired_link_codes()
    code = _generate_link_code()
    pending_link_codes[code] = {
        "uuid": uuid,
        "created_at": time.time(),
    }
    save_link_state()
    return web.json_response({"code": code})


async def create_ticket_api(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    uuid = normalize_text(
        payload.get("uuid")
        or payload.get("player_uuid")
        or payload.get("player")
        or payload.get("user_uuid")
    )
    player_name = normalize_text(
        payload.get("player_name")
        or payload.get("username")
        or payload.get("nick")
        or payload.get("player")
    )
    text = normalize_text(
        payload.get("text")
        or payload.get("message")
        or payload.get("description")
        or payload.get("body")
    )
    kind = normalize_text(payload.get("type") or payload.get("kind"))

    if not text:
        return web.json_response({"error": "empty_text"}, status=400)

    ticket_id = await create_ticket_record(uuid or "legacy", text, kind, player_name)
    return web.json_response({"ticket_id": ticket_id})


async def send_message_api(request: web.Request) -> web.Response:
    ticket_id = normalize_text(request.match_info.get("ticket_id"))
    if not ticket_id:
        return web.json_response({"error": "missing_ticket_id"}, status=400)

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    text = normalize_text(
        payload.get("text")
        or payload.get("message")
        or payload.get("body")
    )
    if not text:
        return web.json_response({"error": "empty_text"}, status=400)

    _append_ticket_message(ticket_id, "player", text)

    topic_id = ticket_topics.get(ticket_id, REPORT_TOPIC_ID)
    try:
        await bot.send_message(
            chat_id=ADMIN_GROUP_ID,
            text=f"<b>Ответ в тикет</b>\nID: <code>{html_escape(ticket_id)}</code>\n\n{html_escape(text)}",
            message_thread_id=topic_id,
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.warning(f"Не удалось переслать сообщение тикета {ticket_id} в Telegram: {e}")

    return web.json_response({"status": "ok"})


async def ask_lemonka_api(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    question = normalize_text(
        payload.get("question")
        or payload.get("text")
        or payload.get("message")
        or payload.get("prompt")
    )
    user_nick = normalize_text(payload.get("player_name") or payload.get("nick") or payload.get("username") or "Игрок")

    if not question:
        return web.json_response({"error": "empty_question"}, status=400)

    try:
        answer = await ask_lemonka_ai(question, user_nick)
    except Exception as e:
        logger.error(f"/lemonka/ask error: {e}")
        return web.json_response({"error": "ai_failed"}, status=500)

    return web.json_response({"answer": answer})


async def observe_chat_api(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    source = normalize_text(payload.get("source") or "unknown")
    room = normalize_text(payload.get("room") or payload.get("chat") or "general")
    speaker = normalize_text(payload.get("speaker") or payload.get("player") or payload.get("user") or "unknown")
    text = normalize_text(payload.get("text") or payload.get("message") or payload.get("content"))

    if not text:
        return web.json_response({"error": "empty_text"}, status=400)

    _remember_chat_message(source, room, speaker, text)
    return web.json_response({"status": "ok"})


async def get_tickets(request: web.Request) -> web.Response:
    uuid = normalize_text(request.query.get("uuid"))
    if not uuid:
        return web.Response(text="")

    ticket_ids = ticket_index.get(uuid, [])
    return web.Response(text="\n".join(ticket_ids))


async def get_messages(request: web.Request) -> web.Response:
    ticket_id = normalize_text(request.query.get("ticket_id") or request.query.get("ticket"))
    if not ticket_id:
        return web.Response(text="")

    return web.Response(text="\n".join(_ticket_history_lines(ticket_id)))


async def close_ticket_api(request: web.Request) -> web.Response:
    ticket_id = normalize_text(request.match_info.get("ticket_id"))
    if not ticket_id:
        return web.json_response({"error": "missing_ticket_id"}, status=400)

    topic_id = ticket_topics.get(ticket_id, REPORT_TOPIC_ID)
    if not close_ticket_record(ticket_id):
        return web.json_response({"error": "ticket_not_found"}, status=404)

    try:
        await bot.send_message(
            chat_id=ADMIN_GROUP_ID,
            text=f"<b>Тикет закрыт</b>\nID: <code>{html_escape(ticket_id)}</code>",
            message_thread_id=topic_id,
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.warning(f"Не удалось отправить сообщение о закрытии тикета {ticket_id}: {e}")

    return web.json_response({"status": "ok"})


async def handle_admin_ticket_message(message: Message) -> bool:
    if message.chat.id != ADMIN_GROUP_ID:
        return False

    topic_id = message.message_thread_id
    if topic_id is None:
        return False

    ticket_id = _find_ticket_by_topic(topic_id)
    if not ticket_id:
        return False

    text = normalize_text(message.text or message.caption)
    if not text or text.startswith("/"):
        return False

    _append_ticket_message(ticket_id, "admin", text)
    return True


# ────────────────────────────────────────────────
# Загрузка китов и знаний
# ────────────────────────────────────────────────

async def load_or_upload_kits():
    global KIT_FILE_IDS
    if os.path.exists(KITS_FILE_IDS_CACHE):
        try:
            with open(KITS_FILE_IDS_CACHE, "r", encoding="utf-8") as f:
                KIT_FILE_IDS = json.load(f)
            logger.info(f"Загружено {len(KIT_FILE_IDS)} file_id китов из кэша")
            if len(KIT_FILE_IDS) >= 5:
                return
        except Exception as e:
            logger.warning(f"Ошибка чтения кэша китов: {e}")

    if not KITS_FOLDER.exists() or not KITS_FOLDER.is_dir():
        logger.warning("Папка 'kits' не найдена!")
        return

    image_files = sorted([f for f in KITS_FOLDER.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif'}])
    if not image_files:
        logger.warning("В папке kits нет картинок")
        return

    logger.info(f"Найдено {len(image_files)} картинок. Загружаем...")
    for file_path in image_files[:5]:
        try:
            sent = await bot.send_photo(OWNER_ID, FSInputFile(file_path), caption=f"Kit upload: {file_path.name}")
            KIT_FILE_IDS[file_path.name] = sent.photo[-1].file_id
            logger.info(f"Загружен {file_path.name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path.name}: {e}")

    if KIT_FILE_IDS:
        try:
            with open(KITS_FILE_IDS_CACHE, "w", encoding="utf-8") as f:
                json.dump(KIT_FILE_IDS, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено {len(KIT_FILE_IDS)} file_id в кэш")
        except Exception as e:
            logger.error(f"Ошибка сохранения кэша китов: {e}")


# ────────────────────────────────────────────────
# Красивое меню
# ────────────────────────────────────────────────

def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(text="🎫 Создать тикет", callback_data="menu_tickets"),
            InlineKeyboardButton(text="🔇 Жалоба на игрока", callback_data="menu_report"),
        ],
        [
            InlineKeyboardButton(text="🤖 ИИ Лемонки", callback_data="menu_ai"),
            InlineKeyboardButton(text="🎁 Киты сервера", callback_data="menu_kits"),
        ],
        [
            InlineKeyboardButton(text="❓ FAQ", callback_data="menu_faq"),
            InlineKeyboardButton(text="🚀 Скачать лаунчер", callback_data="menu_launcher"),
        ],
        [
            InlineKeyboardButton(text="💜 Поддержать сервер", callback_data="menu_donate"),
            InlineKeyboardButton(text="🗣️ Discord", callback_data="menu_discord"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


MENU_TEXT = (
    "<b>ArcaneLegacy:Origin | LemonCraft</b> 🚀\n\n"
    "Привет! Я Ваш помощник на сервере.\n"
    "Выберите, что Вам нужно 👇\n\n"
    "• создать тикет и писать в него как в чате Minecraft\n"
    "• пожаловаться на игрока\n"
    "• спросить ИИ Лемонки\n"
    "• посмотреть наборы привилегий\n"
    "• часто задаваемые вопросы\n"
    "• скачать лаунчер\n"
    "• поддержать проект и получить приятную привилегию\n"
    "• наш уютный Discord-сервер\n\n"
    "<i>Скоро увидимся в игре❤️</i>"
)


@router.callback_query(lambda c: (c.data or "").startswith(("menu_", "ticket_")))
async def process_menu_buttons(callback: CallbackQuery):
    data = callback.data or ""

    if data in {"menu_bug", "menu_tickets"}:
        user_id = callback.from_user.id
        selected_ticket_id = ticket_hub_selection.get(user_id)
        if selected_ticket_id and selected_ticket_id not in get_user_ticket_ids(user_id):
            ticket_hub_selection.pop(user_id, None)
            selected_ticket_id = None

        await callback.message.edit_text(
            build_ticket_hub_text(user_id, selected_ticket_id),
            reply_markup=get_ticket_hub_keyboard(user_id, selected_ticket_id)
        )
    elif data == "menu_home":
        user_id = callback.from_user.id
        ticket_hub_selection.pop(user_id, None)
        pending_actions.pop(callback.from_user.id, None)
        return_to_lemonka.add(user_id)
        await callback.message.edit_text(MENU_TEXT, reply_markup=get_main_menu_keyboard())
    elif data == "menu_report":
        await callback.message.edit_text(
            "Чтобы пожаловаться на игрока, напиши:\n\n"
            "<code>/report @ник причина</code>\n\n"
            "Пример: /report @DragonZX оскорбления\n\n"
            "так-же заскриньте момент нарушения, администратор с вами свяжется для уточнения деталей произошедшего",
            reply_markup=None
        )
    elif data == "menu_ai":
        await callback.message.edit_text(
            "ИИ Лемонки уже тут.\n\n"
            "Просто напиши мне вопрос в личку, например:\n"
            "<code>что такое Apotheosis?</code>\n"
            "<code>как связаться с админами?</code>\n"
            "<code>кто я?</code>",
            reply_markup=None
        )
    elif data == "menu_kits":
        await cmd_kits(callback.message)
        try:
            await callback.message.delete()
        except:
            pass
    elif data == "menu_faq":
        await cmd_faq(callback.message)
    elif data == "menu_donate":
        await cmd_donate(callback.message)
    elif data == "menu_discord":
        await cmd_discord(callback.message)
    elif data == "menu_launcher":
        await callback.message.edit_text(
            f"<b>Скачай лаунчер LemonCraft прямо сейчас!</b>\n\n"
            f"Переходи по ссылке:\n"
            f"<a href=\"{LEMON_LINK}\">🔗 {LEMON_LINK}</a>\n\n"
            "После скачивания просто запусти лаунчер и войди в аккаунт — добро пожаловать на сервер! 🎮",
            reply_markup=None,
            disable_web_page_preview=False
        )
    elif data.startswith("ticket_select:"):
        user_id = callback.from_user.id
        ticket_id = data.split(":", 1)[1].strip()
        if ticket_id:
            ticket_hub_selection[user_id] = ticket_id
            await callback.message.edit_text(
                build_ticket_hub_text(user_id, ticket_id),
                reply_markup=get_ticket_hub_keyboard(user_id, ticket_id)
            )
    elif data == "ticket_new":
        user_id = callback.from_user.id
        pending_actions[user_id] = {"mode": "create_ticket"}
        ticket_hub_selection.pop(user_id, None)
        await callback.message.edit_text(
            "<b>Создание нового тикета</b>\n\n"
            "Напиши первое сообщение тикета прямо в этот чат.\n"
            "Можно просто отправить текстом, и я создам тикет автоматически.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Вернуться к тикетам", callback_data="menu_tickets")],
                [InlineKeyboardButton(text="Лемонка", callback_data="menu_home")],
            ])
        )

    await callback.answer()


# ────────────────────────────────────────────────
# Команды
# ────────────────────────────────────────────────

@router.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(
        f"Chat ID: <code>{message.chat.id}</code>\n"
        f"Thread ID: <code>{message.message_thread_id or 0}</code>",
        parse_mode=ParseMode.HTML
    )


@router.message(Command("link"))
async def cmd_link(message: Message, command: CommandObject):
    args = (command.args or "").strip()
    if not args:
        current_uuid = linked_accounts.get(message.from_user.id)
        if current_uuid:
            await message.answer(
                f"Аккаунт уже привязан к UUID:\n<code>{html_escape(current_uuid)}</code>\n\n"
                "Если хочешь перепривязать, сначала отвяжи старый аккаунт вручную."
            )
        else:
            await message.answer(
                "Напиши код привязки, который выдал Minecraft-клиент:\n\n"
                "<code>/link 123456</code>"
            )
        return

    code = args.split(maxsplit=1)[0]
    cleanup_expired_link_codes()
    link_info = pending_link_codes.get(code)
    if not link_info:
        await message.answer("Код привязки не найден или уже устарел.")
        return

    uuid = normalize_text(link_info.get("uuid"))
    if not uuid:
        await message.answer("Код привязки не найден или уже устарел.")
        return

    linked_accounts[message.from_user.id] = uuid
    pending_link_codes.pop(code, None)
    save_link_state()

    await message.answer(
        "Аккаунт успешно привязан.\n\n"
        f"UUID: <code>{html_escape(uuid)}</code>\n"
        "Теперь твои тикеты из Minecraft появятся в меню, а новые тикеты будут общими."
    )


@router.message(Command("unlink"))
async def cmd_unlink(message: Message):
    if message.from_user.id not in linked_accounts:
        await message.answer("Аккаунт еще не привязан.")
        return

    linked_accounts.pop(message.from_user.id, None)
    save_link_state()
    await message.answer("Привязка удалена.")


@router.message(Command("menu"))
async def cmd_menu(message: Message):
    await message.answer(MENU_TEXT, reply_markup=get_main_menu_keyboard())


@router.message(Command("reg"))
async def cmd_reg(message: Message, command: CommandObject):
    args = (command.args or "").strip()
    if not args:
        current_nick = nicknames.get(message.from_user.id, None)
        if current_nick:
            await message.answer(
                f"Твой текущий ник: <b>{current_nick}</b>\n\n"
                f"Чтобы изменить — напиши /reg НовыйНик"
            )
        else:
            await message.answer(
                "Укажи свой ник на сервере:\n\n"
                "<code>/reg ТвойНик</code>\n\n"
                "Например: /reg DragonZX"
            )
        return

    new_nick = args
    tg_id = message.from_user.id
    tg_username = message.from_user.username or "не указан"

    nicknames[tg_id] = new_nick
    save_nicknames()

    if tg_id not in players:
        players[tg_id] = {}

    players[tg_id].update({
        "server_nick": new_nick,
        "tg_username": f"@{tg_username}" if tg_username != "не указан" else "не указан",
        "home_points": players[tg_id].get("home_points", "не установлено"),
        "base_coords": players[tg_id].get("base_coords", "не установлено")
    })
    save_players()
    load_ai_data()

    await message.answer(
        f"Отлично! Теперь я знаю тебя как <b>{new_nick}</b> ❤️\n"
        f"Твои данные добавлены в базу игроков.\n"
        f"Можешь писать репорты и вопросы — я учту твой ник~ ♡"
    )


@router.message(Command("info"))
async def cmd_info(message: Message, command: CommandObject):
    if message.chat.id != ADMIN_GROUP_ID or message.message_thread_id != REPORT_TOPIC_ID:
        await message.answer("Эта команда доступна только в теме #21971 в админ-чате.")
        return

    args = (command.args or "").strip()
    if not args:
        await message.answer("Укажи ник игрока:\n\n<code>/info НикНаСервере</code>")
        return

    target_nick = args.lower()
    found = False

    for tg_id, player_data in players.items():
        server_nick = str(player_data.get("server_nick", "")).lower()
        if server_nick == target_nick:
            tg_username = player_data.get("tg_username", "не указан")
            home_points = player_data.get("home_points", "не установлено")
            base_coords = player_data.get("base_coords", "не установлено")

            text = (
                f"<b>Инфо по нику {player_data.get('server_nick')}</b>\n\n"
                f"1. Тег в ТГ: {tg_username} [{tg_id}]\n"
                f"2. Ник на сервере: <b>{player_data.get('server_nick')}</b>\n"
                f"3. Точки дома: <i>{home_points}</i>\n"
                f"4. Координаты базы: <i>{base_coords}</i>"
            )
            await message.answer(text)
            found = True
            break

    if not found:
        await message.answer(f"Игрок с ником **{args}** не найден в базе.")


@router.message(Command("delnick"))
async def cmd_delnick(message: Message, command: CommandObject):
    if message.chat.id != ADMIN_GROUP_ID or message.message_thread_id != REPORT_TOPIC_ID:
        await message.answer("Эта команда доступна только в теме #21971 в админ-чате.")
        return

    args = (command.args or "").strip()
    if not args.startswith("@"):
        await message.answer("Укажи тег: /delnick @ник")
        return

    target_username = args[1:].lower()
    found = False

    for tg_id in list(players.keys()):
        try:
            user = await bot.get_chat(tg_id)
            if user.username and user.username.lower() == target_username:
                if tg_id in nicknames:
                    del nicknames[tg_id]
                del players[tg_id]
                save_nicknames()
                save_players()
                load_ai_data()
                await message.answer(f"Игрок @{user.username} удалён из базы")
                found = True
                break
        except Exception:
            continue

    if not found:
        await message.answer(f"@{target_username} не найден.")


@router.message(Command("closeticket", "close_ticket"))
async def cmd_closeticket(message: Message, command: CommandObject):
    if message.chat.id != ADMIN_GROUP_ID:
        await message.answer("Эта команда доступна только в админ-чате.")
        return

    ticket_id = ""
    args = (command.args or "").strip()
    if args:
        ticket_id = args.split(maxsplit=1)[0].strip()
    elif message.message_thread_id is not None:
        ticket_id = _find_ticket_by_topic(message.message_thread_id)

    if not ticket_id:
        await message.answer(
            "Укажи ID тикета или отправь команду в теме тикета.\n"
            "Пример: /closeticket 123456789"
        )
        return

    topic_id = ticket_topics.get(ticket_id, message.message_thread_id or REPORT_TOPIC_ID)
    if not close_ticket_record(ticket_id):
        await message.answer(f"Тикет <code>{html_escape(ticket_id)}</code> не найден.")
        return

    await bot.send_message(
        chat_id=ADMIN_GROUP_ID,
        text=f"<b>Тикет закрыт</b>\nID: <code>{html_escape(ticket_id)}</code>",
        message_thread_id=topic_id,
        disable_web_page_preview=True
    )

    await message.answer(f"Тикет <code>{html_escape(ticket_id)}</code> закрыт.")


@router.message(Command("lemonka", "ai", "ask"))
async def cmd_lemonka(message: Message, command: CommandObject):
    question = (command.args or "").strip()
    if not question:
        await message.answer(
            "Напиши вопрос так:\n"
            "<code>/lemonka что такое Apotheosis?</code>\n"
            "или просто отправь сообщение в личку боту.",
            parse_mode=ParseMode.HTML
        )
        return

    try:
        user_nick = nicknames.get(message.from_user.id, message.from_user.first_name)
        answer = await ask_lemonka_ai(question, user_nick)
        await message.answer(html_escape(answer), disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка ArtemoX API: {e}")
        await message.answer("Оййй... Лемонка чуток задумалась~ ♡ Попробуй ещё разок, мой хороший? 💛")


@router.message(Command("reload_lemonka"))
async def cmd_reload_lemonka(message: Message):
    is_owner = message.from_user.id == OWNER_ID
    is_in_admin_thread = (message.chat.id == ADMIN_GROUP_ID and message.message_thread_id == REPORT_TOPIC_ID)

    if not (is_owner or is_in_admin_thread):
        return

    if is_in_admin_thread:
        await bot.send_message(
            chat_id=ADMIN_GROUP_ID,
            text="Перезагружаю Лемонку полностью... ♡ Подожди 5–10 секунд~ 💛",
            message_thread_id=REPORT_TOPIC_ID
        )
    else:
        await message.answer("Перезагружаю Лемонку полностью... ♡ Подожди 5–10 секунд~ 💛")

    logger.info("Команда /reload_lemonka: полная перезагрузка бота инициирована")
    await dp.stop_polling()
    python = sys.executable
    os.execl(python, python, *sys.argv)


@router.message(Command("reload_knowledge"))
async def cmd_reload_knowledge(message: Message):
    if message.from_user.id != OWNER_ID:
        await message.answer("Только владелец может обновлять знания~")
        return

    load_ai_data()
    await message.answer("База знаний обновлена! ♡ Эмбеддинги пересчитаны~ 💛")


@router.message(Command("bug"))
async def cmd_bug(message: Message, command: CommandObject):
    description = (command.args or "").strip()
    if len(description) < 10:
        await message.answer("Опиши баг подробнее (минимум 10 символов). Пример: /bug Не заходит на сервер, ошибка 404")
        return

    user = message.from_user
    text = f"<b>Баг-репорт</b>\nОт: {user.full_name} (@{user.username or 'нет'}) [{user.id}]\n\n{description}"

    try:
        media = []
        if message.photo: media = [InputMediaPhoto(media=p.file_id) for p in message.photo]
        elif message.video: media = [InputMediaVideo(media=message.video.file_id)]
        elif message.document: media = [InputMediaDocument(media=message.document.file_id)]

        if media:
            await bot.send_media_group(ADMIN_GROUP_ID, media=media, caption=text, message_thread_id=REPORT_TOPIC_ID)
        else:
            await bot.send_message(ADMIN_GROUP_ID, text, message_thread_id=REPORT_TOPIC_ID)
        await message.answer("Баг отправлен Администраторам!\nС вами может связаться один из Администраторов для уточнения деталей")
    except Exception as e:
        logger.error(f"/bug error: {e}")
        await message.answer("Ошибка отправки. Попробуй позже.")


@router.message(Command("report"))
async def cmd_report_user(message: Message, command: CommandObject):
    args = (command.args or "").strip().split(maxsplit=1)
    if not args or not args[0].startswith('@'):
        await message.answer("Использование: /report @ник [причина]")
        return
    target = args[0]
    reason = args[1] if len(args) > 1 else "причина не указана"

    text = f"<b>Жалоба на игрока</b>\nОт: {message.from_user.full_name} [{message.from_user.id}]\nНа кого: {target}\nПричина: {reason}"

    try:
        media = []
        if message.photo:
            media = [InputMediaPhoto(media=p.file_id) for p in message.photo]
        elif message.video:
            media = [InputMediaVideo(media=message.video.file_id)]
        elif message.document:
            media = [InputMediaDocument(media=message.document.file_id)]

        if media:
            await bot.send_media_group(ADMIN_GROUP_ID, media=media, caption=text, message_thread_id=REPORT_TOPIC_ID)
        else:
            await bot.send_message(ADMIN_GROUP_ID, text, message_thread_id=REPORT_TOPIC_ID)
        await message.answer(f"Жалоба на {target} отправлена Администраторам.")
    except Exception as e:
        logger.error(f"/report error: {e}")
        await message.answer("Ошибка отправки. Попробуй позже.")


@router.message(Command("discord"))
async def cmd_discord(message: Message):
    await message.answer(f"Наш Лимонный Discord: {DISCORD_LINK}")


@router.message(Command("donate"))
async def cmd_donate(message: Message):
    await message.answer(f"<b>Поддержи сервер и получи за это уникальные возможности!</b>\nПривилегии и киты: {DONATE_LINK}")


@router.message(Command("kits", "kit"))
async def cmd_kits(message: Message):
    if len(KIT_FILE_IDS) == 0:
        await load_or_upload_kits()

    if len(KIT_FILE_IDS) == 0:
        await message.answer("Картинки китов не найдены. Проверь папку 'kits' и логи.")
        return

    kit_captions = KIT_CAPTIONS
    file_ids = list(KIT_FILE_IDS.values())[:5]

    try:
        for i, fid in enumerate(file_ids):
            caption = kit_captions[i] if i < len(kit_captions) else f"Кит №{i+1}"
            await bot.send_photo(
                chat_id=message.chat.id,
                photo=fid,
                caption=f"<b>{caption}</b>",
                parse_mode=ParseMode.HTML
            )
            await asyncio.sleep(0.4)

    except Exception as e:
        logger.error(f"/kits error: {e}")
        await message.answer("Не удалось отправить киты. Попробуй позже.")


@router.message(Command("faq"))
async def cmd_faq(message: Message):
    faq_text = f"""<b>❓ Часто задаваемые вопросы</b>

1. Как зайти на сервер?
Скачать наш <a href="{LEMON_LINK}">лаунчер (кликабельно)</a>

2. Как получить киты?
можете прописать /kits и посмотреть все доступные вам наборы, либо зайти в главное меню и нажать кнопочку "киты"

3. Где купить привилегию?
<a href="{DONATE_LINK}">Магазин привелегий</a>

4. Хочу отправить жалобу, как это сделать?
Используй /bug для багов или /report @ник для жалоб

5. Есть ли Discord?
<a href="{DISCORD_LINK}">Заходи сюда</a>

<i>Если не нашли ответа на свой вопрос - вы можете написать одному из Администраторов (Bishepp, PredatorX, DragonZX), Администратор ответит вам как только освободится</i>"""

    await message.answer(faq_text, disable_web_page_preview=True)


@router.message()
async def handle_message(message: Message):
    if message.chat.id == ADMIN_GROUP_ID:
        await handle_admin_ticket_message(message)
        return

    # ────────────────────────────────────────────────
    # Лемонка учится на ВСЕХ чатах и личных сообщениях
    # ────────────────────────────────────────────────
    text = normalize_text(message.text or message.caption)
    if text and not text.startswith("/"):
        speaker = message.from_user.full_name if message.from_user else "unknown"
        
        await observe_chat_message(
            "telegram",
            str(message.chat.id),      # ID чата (группа или личка)
            speaker,
            text,
        )

    # Если сообщение в группе — только учимся, не отвечаем
    if message.chat.type in ("group", "supergroup"):
        return

    # Если не приватный чат — выходим
    if message.chat.type != "private":
        return

    # ────────────────────────────────────────────────
    # Ниже оригинальная логика для личных сообщений
    # ────────────────────────────────────────────────
    if not message.text or message.text.startswith("/"):
        return

    user_id = message.from_user.id
    if user_id in return_to_lemonka:
        return_to_lemonka.discard(user_id)
        ticket_hub_selection.pop(user_id, None)

    pending = pending_actions.get(user_id, {})
    if pending.get("mode") == "create_ticket":
        ticket_text = normalize_text(message.text or message.caption)
        if not ticket_text:
            await message.answer("Отправь текст первого сообщения тикета.")
            return

        owner_uuid = linked_accounts.get(user_id) or get_telegram_ticket_uuid(user_id)
        ticket_id = await create_ticket_record(
            owner_uuid,
            ticket_text,
            "telegram",
            message.from_user.full_name,
        )
        pending_actions.pop(user_id, None)
        if ticket_id:
            ticket_hub_selection[user_id] = ticket_id
            await message.answer(
                "Тикет создан. Теперь можешь писать в него дальше.",
                reply_markup=get_ticket_hub_keyboard(user_id, ticket_id)
            )
        return

    selected_ticket_id = ticket_hub_selection.get(user_id)
    if selected_ticket_id and selected_ticket_id in ticket_messages:
        ticket_text = normalize_text(message.text or message.caption)
        if ticket_text:
            await send_ticket_message_to_admin(
                selected_ticket_id,
                message.from_user.full_name,
                user_id,
                ticket_text,
            )
            await message.answer(
                f"Сообщение отправлено в тикет <code>{html_escape(selected_ticket_id)}</code>.",
                reply_markup=get_ticket_hub_keyboard(user_id, selected_ticket_id)
            )
            return

    await forward_to_owner(message)

    user_nick = nicknames.get(message.from_user.id, message.from_user.first_name)

    lemonka_reply = None
    try:
        lemonka_reply = await ask_lemonka_ai(message.text, user_nick)
        await message.answer(html_escape(lemonka_reply), disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка ArtemoX API: {e}")
        lemonka_reply = "Оййй... Лемонка чуток задумалась~ ♡ Попробуй ещё разок, мой хороший? 💛"
        await message.answer(lemonka_reply)

    # Запоминаем ответ самой Лемонки (чтобы она училась на своих репликах)
    if lemonka_reply:
        await observe_chat_message(
            "telegram",
            str(message.chat.id),
            "Лемонка",
            lemonka_reply
        )


async def forward_to_owner(message: Message):
    user = message.from_user
    text = message.text or message.caption or ""
    caption = f"<b>Сообщение от игрока</b>\nОт: {user.full_name} (@{user.username or 'нет'}) [{user.id}]\n\n{text}"
    try:
        if message.photo:
            await bot.send_photo(OWNER_ID, message.photo[-1].file_id, caption=caption)
        elif message.video:
            await bot.send_video(OWNER_ID, message.video.file_id, caption=caption)
        elif message.document:
            await bot.send_document(OWNER_ID, message.document.file_id, caption=caption)
        else:
            await bot.send_message(OWNER_ID, caption, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка отправки владельцу: {e}")


async def start_web() -> None:
    global web_runner

    app = web.Application()
    app.router.add_post("/link", create_link)
    app.router.add_post("/ticket", create_ticket_api)
    app.router.add_post("/ticket/{ticket_id}", send_message_api)
    app.router.add_post("/ticket/{ticket_id}/close", close_ticket_api)
    app.router.add_post("/lemonka/ask", ask_lemonka_api)
    app.router.add_post("/lemonka/observe", observe_chat_api)
    app.router.add_get("/tickets", get_tickets)
    app.router.add_get("/messages", get_messages)

    web_runner = web.AppRunner(app)
    await web_runner.setup()

    site = web.TCPSite(web_runner, "0.0.0.0", 8080)
    await site.start()
    logger.info("HTTP API запущен на 0.0.0.0:8080")


async def stop_web() -> None:
    global web_runner

    if web_runner is not None:
        await web_runner.cleanup()
        web_runner = None


async def main():
    logger.info("Бот запущен")
    load_ai_data()
    load_ticket_state()
    load_link_state()
    await load_or_upload_kits()
    await start_web()
    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        await stop_web()


if __name__ == "__main__":
    asyncio.run(main())