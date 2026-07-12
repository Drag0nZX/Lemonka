import asyncio
import logging
import json
import os
import random
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
SUMMARY_TELEGRAM_CHAT_IDS = {2403528036}
ADMIN_IDS: set[int] = {
    OWNER_ID,
    6248897312,
    295735470,
    587927728
}

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
DIALOGUE_STYLE_FILE = Path("dialogue_style.json")
RELOAD_NOTICE_FILE = Path("reload_notice.json")
MODERATION_RULES_FILE = Path("moderation_rules.txt")
MODERATION_STRIKES_FILE = Path("moderation_strikes.json")
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
dialogue_style: dict[str, dict[str, Any]] = {}
reload_notice: dict[str, Any] = {}
moderation_rules_text: str = ""
moderation_strikes: dict[str, dict[str, int]] = {}
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
    timeout=60.0,
    max_retries=2
)

LEMONKA_AI_TIMEOUT_SECONDS = 120
LEMONKA_AI_MAX_TOKENS = 800
LEMONKA_AI_CHAT_CONTEXT_LIMIT = 8
LEMONKA_AI_DIALOG_HISTORY_LIMIT = 10

# Подписи для китов (в порядке файлов в папке)
KIT_CAPTIONS = [
    "VIP набор",
    "Premium набор",
    "Diamond набор",
    "Koya набор",
    "Deluxe набор"
    "Bonus набор"
    "Smelting набор"
    "Magic набор"
    "Gems набор"
    "storage набор"
    "Custom набор"
    "DeluxeCustom набор"
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


def load_dialogue_style() -> None:
    global dialogue_style
    data = load_json(DIALOGUE_STYLE_FILE, {})
    if not isinstance(data, dict):
        dialogue_style = {}
        return

    cleaned: dict[str, dict[str, Any]] = {}
    for actor_key, info in data.items():
        if not isinstance(info, dict):
            continue
        cleaned[str(actor_key)] = {
            "messages": int(info.get("messages") or 0),
            "casual": int(info.get("casual") or 0),
            "technical": int(info.get("technical") or 0),
            "warm": int(info.get("warm") or 0),
            "last": [normalize_text(item) for item in (info.get("last") or []) if normalize_text(item)],
        }
    dialogue_style = cleaned


def save_dialogue_style() -> None:
    try:
        with open(DIALOGUE_STYLE_FILE, "w", encoding="utf-8") as f:
            json.dump(dialogue_style, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения dialogue style: {e}")


def load_reload_notice() -> None:
    global reload_notice
    data = load_json(RELOAD_NOTICE_FILE, {})
    reload_notice = data if isinstance(data, dict) else {}


def save_reload_notice() -> None:
    try:
        with open(RELOAD_NOTICE_FILE, "w", encoding="utf-8") as f:
            json.dump(reload_notice, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения reload notice: {e}")


def clear_reload_notice() -> None:
    global reload_notice
    reload_notice = {}
    try:
        if RELOAD_NOTICE_FILE.exists():
            RELOAD_NOTICE_FILE.unlink()
    except Exception as e:
        logger.warning(f"Не удалось удалить reload notice: {e}")


def _ensure_dialogue_style(actor_key: str) -> dict[str, Any]:
    actor_key = normalize_actor_key(actor_key)
    if not actor_key:
        return {}
    bucket = dialogue_style.setdefault(actor_key, {
        "messages": 0,
        "casual": 0,
        "technical": 0,
        "warm": 0,
        "last": [],
    })
    bucket.setdefault("messages", 0)
    bucket.setdefault("casual", 0)
    bucket.setdefault("technical", 0)
    bucket.setdefault("warm", 0)
    bucket.setdefault("last", [])
    return bucket


def learn_dialogue_style(actor_key: str, text: str) -> None:
    actor_key = normalize_actor_key(actor_key)
    text = normalize_text(text)
    if not actor_key or not text:
        return

    lower = text.lower()
    bucket = _ensure_dialogue_style(actor_key)
    if not bucket:
        return

    bucket["messages"] = int(bucket.get("messages", 0)) + 1

    casual_markers = (
        "привет", "ку", "хай", "hello", "hi", "спасибо", "спс", "пж", "пожалуй",
        "как дела", "как ты", "что делаешь", "чем занят", "пока", "доброе утро",
        "добрый день", "добрый вечер", "ладно", "угу", "ага", "ок", "окей"
    )
    technical_markers = (
        "команда", "ошибка", "краш", "лаг", "инстанс", "клиент", "сервер", "uuid",
        "тикет", "репорт", "лог", "конфиг", "mod", "мод", "api", "prompt", "промпт"
    )
    warm_markers = ("❤️", "♡", "💕", "💛", "💜", "😘", "мила", "солн", "котик", "зайк", "мой хороший")

    if any(marker in lower for marker in casual_markers):
        bucket["casual"] = int(bucket.get("casual", 0)) + 1
    if any(marker in lower for marker in technical_markers):
        bucket["technical"] = int(bucket.get("technical", 0)) + 1
    if any(marker in lower for marker in warm_markers):
        bucket["warm"] = int(bucket.get("warm", 0)) + 1

    last_items = bucket.setdefault("last", [])
    last_items.append(text)
    if len(last_items) > 16:
        del last_items[:-16]

    save_dialogue_style()


def get_dialogue_style_profile(actor_key: str) -> dict[str, Any]:
    actor_key = normalize_actor_key(actor_key)
    if not actor_key:
        return {"mode": "neutral", "warm": 0, "casual": 0, "technical": 0}

    bucket = dialogue_style.get(actor_key) or {}
    casual = int(bucket.get("casual", 0) or 0)
    technical = int(bucket.get("technical", 0) or 0)
    warm = int(bucket.get("warm", 0) or 0)

    mode = "neutral"
    if casual > technical:
        mode = "casual"
    elif technical > casual:
        mode = "technical"

    return {
        "mode": mode,
        "warm": warm,
        "casual": casual,
        "technical": technical,
    }


async def is_admin_user(message: Message) -> bool:
    if not message.from_user:
        return False

    if message.from_user.id in ADMIN_IDS or message.from_user.id == OWNER_ID:
        return True

    try:
        member = await bot.get_chat_member(ADMIN_GROUP_ID, message.from_user.id)
        return member.status in ("administrator", "creator")
    except Exception:
        return False


def humanize_lemonka_reply(answer: str, actor_key: str, user_text: str = "") -> str:
    answer = normalize_text(answer)
    if not answer:
        return answer

    profile = get_dialogue_style_profile(actor_key)
    lower = answer.lower()

    opener_replacements = [
        ("конечно,", "ага,"),
        ("конечно!", "ага!"),
        ("да,", "угу,"),
        ("да!", "угу!"),
        ("нет,", "не-а,"),
        ("нет!", "не-а!"),
        ("хм,", "мм,"),
        ("хмм,", "мм,"),
        ("вот,", "смотри,"),
        ("вот что", "смотри, что"),
    ]
    for old, new in opener_replacements:
        if lower.startswith(old):
            answer = new + answer[len(old):]
            lower = answer.lower()
            break

    answer = answer.replace("!!", "!").replace("??", "?").replace("  ", " ").strip()

    rng = random.Random(f"{actor_key}|{user_text}|{answer}")

    if profile["mode"] == "casual":
        soft_openers = ["Ой, ", "Мм, ", "Ага, ", "Угу, ", "Солнышко, "]
        if len(answer) < 220 and not answer.startswith(tuple(soft_openers)) and rng.random() < 0.38:
            answer = rng.choice(soft_openers) + answer[0].lower() + answer[1:] if answer else answer
    elif profile["mode"] == "technical":
        technical_openers = ["Смотри, ", "По сути, ", "Да, ", "Если коротко, "]
        if len(answer) < 240 and not answer.startswith(tuple(technical_openers)) and rng.random() < 0.22:
            answer = rng.choice(technical_openers) + answer[0].lower() + answer[1:] if answer else answer

    if profile["warm"] > 3 and "♡" not in answer and "❤️" not in answer and rng.random() < 0.25:
        answer = f"{answer} ♡"

    if len(answer) > 3600:
        answer = answer[:3597].rstrip() + "..."

    return answer


def get_recent_public_chat_lines(hours: int = 8, limit: int = 240) -> list[str]:
    cutoff = time.time() - max(1, hours) * 3600
    lines: list[str] = []

    for item in chat_memory:
        ts = float(item.get("ts") or 0)
        if ts < cutoff:
            continue

        source = normalize_text(item.get("source")).lower()
        room = normalize_text(item.get("room"))
        speaker = normalize_text(item.get("speaker"))
        text = normalize_text(item.get("text"))
        if not text:
            continue

        is_public_mc = source == "minecraft" and room == "public"
        is_public_tg = source == "telegram" and _telegram_room_matches_summary(room)

        if not (is_public_mc or is_public_tg):
            continue

        stamp = time.strftime("%H:%M", time.localtime(ts))
        lines.append(f"[{stamp}] {source}/{room} {speaker}: {text}")

    if len(lines) > limit:
        lines = lines[-limit:]
    return lines


def build_local_chat_summary(lines: list[str], hours: int) -> str:
    if not lines:
        return "За последние часы я почти ничего не увидела в общих чатах, мой хороший."

    categories = [
        ("Гриферство и защита", "обсуждали гриферов, разрушения, кражи, рейды и защиту базы или дома", ("грифер", "гриф", "гриферство", "рейд", "разруш", "ломал", "сломал", "снес", "воров", "краж", "украл", "порча", "сжег")),
        ("Техпроблемы и баги", "обсуждали лаги, ошибки, вылеты и то, как ведёт себя сервер или клиент", ("лаг", "краш", "ошиб", "баг", "пинг", "вылет", "завис", "тормоз", "не работает", "не заходит")),
        ("Тикеты и поддержка", "спрашивали про тикеты, жалобы и помощь админов", ("тикет", "поддерж", "репорт", "жалоб", "обращен", "помог", "админ")),
        ("Игровые дела", "говорили про спавн, ивенты, квесты, фарм, постройки и развитие в мире", ("сервер", "мир", "спавн", "ивент", "квест", "босс", "фарм", "ресурс", "стро", "пвп", "выживание")),
        ("Привязка и связь", "разбирали привязку аккаунта, Discord/Telegram и общение в ЛС", ("link", "/link", "привяз", "дискорд", "телеграм", "лс", "личк")),
        ("Обычное общение", "просто здоровались, отвечали друг другу и обсуждали повседневные вещи вроде 'как дела' и 'что делаешь'", ("привет", "ку", "здрав", "спасибо", "пока", "как дела", "как ты", "что делаешь", "общ", "давай")),
        ("Донат и киты", "спрашивали про киты, привилегии и магазин", ("донат", "привил", "кит", "набор", "магазин")),
    ]

    counts: list[tuple[int, str, str]] = []
    lower_lines = [line.lower() for line in lines]
    for title, description, keywords in categories:
        matches = 0
        for line, lower in zip(lines, lower_lines):
            if any(keyword in lower for keyword in keywords):
                matches += 1
        if matches:
            counts.append((matches, title, description))

    counts.sort(key=lambda item: item[0], reverse=True)
    non_generic = [item for item in counts if item[1] != "Обычное общение"]
    selected = (non_generic or counts)[:4]

    if not selected:
        source_counts: dict[str, int] = {"minecraft": 0, "telegram": 0}
        for line in lines:
            if "/minecraft/" in line:
                source_counts["minecraft"] += 1
            elif "/telegram/" in line:
                source_counts["telegram"] += 1

        parts = []
        if source_counts["minecraft"]:
            parts.append(f"Minecraft: {source_counts['minecraft']} сообщений")
        if source_counts["telegram"]:
            parts.append(f"Telegram: {source_counts['telegram']} сообщений")

        if parts:
            return (
                f"За последние {hours} часов в общих чатах было немного сообщений. "
                f"В основном это обычное общение игроков, вопросы и короткие ответы. "
                f"По площадкам: {', '.join(parts)}."
            )

        return (
            f"За последние {hours} часов в общих чатах было обычное игровое общение, "
            f"вопросы и короткие ответы, без ярко выраженной одной темы."
        )

    if len(selected) == 1 and selected[0][1] == "Обычное общение":
        return (
            f"За последние {hours} часов игроки в основном просто общались между собой: "
            f"здоровались, спрашивали 'как дела', благодарили друг друга и перебрасывались короткими бытовыми фразами. "
            f"Это был спокойный, повседневный чат без большой общей темы."
        )

    if selected and all(title == "Обычное общение" for _, title, _ in selected):
        return (
            f"За последние {hours} часов игроки в основном просто общались между собой: "
            f"здоровались, обменивались короткими ответами и обсуждали бытовые мелочи. "
            f"Это был спокойный повседневный чат."
        )

    if selected and selected[0][1] == "Гриферство и защита":
        return (
            f"За последние {hours} часов игроки больше всего обсуждали гриферство и защиту построек: "
            f"жалобы на разрушения, кражи, рейды и то, как обезопасить дом или базу. "
            f"Если совсем кратко: разговор шёл про вредителей и способы защититься от них."
        )

    bullets = []
    for count, title, description in selected:
        if count == 1:
            bullets.append(f"• {title}: {description}.")
        else:
            bullets.append(f"• {title}: {description}. Это всплывало {count} раз.")

    top_title = selected[0][1]
    top_description = selected[0][2]
    total = len(lines)
    intro = (
        f"За последние {hours} часов в общих чатах было {total} сообщений. "
        f"Главная мысль диалогов: {top_description}."
    )
    return "\n".join([intro, *bullets[:3]])


def _parse_summary_line_v2(line: str) -> dict[str, str] | None:
    line = normalize_text(line)
    if not line.startswith("[") or "] " not in line:
        return None

    stamp, rest = line[1:].split("] ", 1)
    if len(stamp) != 5 or ":" not in stamp:
        return None

    meta, sep, text = rest.partition(": ")
    if not sep:
        return None

    if " " in meta:
        source_room, speaker = meta.split(" ", 1)
    else:
        source_room, speaker = meta, ""

    if "/" in source_room:
        source, room = source_room.split("/", 1)
    else:
        source, room = source_room, ""

    return {
        "stamp": stamp,
        "source": source,
        "room": room,
        "speaker": speaker,
        "text": text,
        "lower": text.lower(),
    }


def _format_summary_range_v2(stamps: list[str]) -> str:
    if not stamps:
        return "примерно в это время"
    if len(stamps) == 1:
        return stamps[0]
    return f"{stamps[0]}-{stamps[-1]}"


def _shorten_summary_text(text: str, limit: int = 11) -> str:
    words = normalize_text(text).split()
    if not words:
        return ""
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]) + "…"


def build_local_chat_summary(lines: list[str], hours: int) -> str:
    if not lines:
        return f"За последние {hours} часов в общих чатах почти ничего не писали."

    parsed_lines = [item for item in (_parse_summary_line_v2(line) for line in lines) if item]
    if not parsed_lines:
        return f"За последние {hours} часов в общих чатах почти ничего не писали."

    topic_rules = [
        {
            "title": "Гриферство и защита",
            "keywords": ("грифер", "рейд", "разруш", "ломал", "сломал", "снес", "крад", "украл", "порч", "сжег"),
            "details": [
                ("жаловались на разрушения, кражи и рейды", ("разруш", "ломал", "сломал", "снес", "крад", "украл", "порч", "сжег", "рейд")),
                ("обсуждали, кто это сделал и где ударили по базе", ("кто", "винов", "подозр", "баз", "дом", "домик")),
                ("думали, как защитить постройки и не дать повторить атаку", ("защит", "укреп", "щит", "замок", "ворот", "баз", "дом")),
            ],
        },
        {
            "title": "Техпроблемы и баги",
            "keywords": ("лаг", "краш", "ошиб", "баг", "пинг", "вылет", "завис", "тормоз", "не работает", "не заходит"),
            "details": [
                ("обсуждали лаги, ошибки и вылеты", ("лаг", "краш", "ошиб", "баг", "пинг", "вылет", "завис", "тормоз")),
                ("пытались понять, что ломает клиент или сервер", ("не работает", "не заходит", "сервер", "клиент", "слом", "проблем")),
            ],
        },
        {
            "title": "Тикеты и поддержка",
            "keywords": ("тикет", "поддерж", "репорт", "жалоб", "обращен", "помог", "админ"),
            "details": [
                ("спрашивали, как открыть тикет и куда писать проблему", ("тикет", "репорт", "жалоб", "обращен", "помог")),
                ("разбирали ответы админов и дальнейшие действия", ("админ", "модер", "поддерж", "ответ", "жалоб")),
            ],
        },
        {
            "title": "Игровые дела",
            "keywords": ("сервер", "мир", "спавн", "ивент", "квест", "босс", "фарм", "ресурс", "стро", "пвп", "выживание"),
            "details": [
                ("говорили про спавн, ивенты, квесты и фарм", ("спавн", "ивент", "квест", "босс", "фарм", "ресурс", "пвп")),
                ("обсуждали постройки, развитие мира и игровые планы", ("стро", "мир", "выжив", "развит", "план")),
            ],
        },
        {
            "title": "Привязка и связь",
            "keywords": ("link", "/link", "привяз", "дискорд", "телеграм"),
            "details": [
                ("настраивали привязку аккаунта и связь между игрой и Telegram", ("link", "/link", "привяз", "телеграм", "дискорд")),
            ],
        },
        {
            "title": "Донат и киты",
            "keywords": ("донат", "привил", "кит", "набор", "магазин"),
            "details": [
                ("спрашивали про донат, киты и привилегии", ("донат", "привил", "кит", "набор")),
                ("уточняли, что дают наборы или магазин", ("магазин", "донат", "кит")),
            ],
        },
        {
            "title": "Обычное общение",
            "keywords": ("привет", "ку", "здрав", "спасибо", "пока", "как дела", "как ты", "что делаешь", "общ"),
            "details": [
                ("здоровались и перекидывались короткими бытовыми фразами", ("привет", "ку", "здрав", "спасибо", "пока", "как дела", "как ты", "что делаешь")),
            ],
        },
    ]

    snapshots: list[dict[str, object]] = []
    for topic in topic_rules:
        matched = [item for item in parsed_lines if any(keyword in item["lower"] for keyword in topic["keywords"])]
        if not matched:
            continue

        stamps = [item["stamp"] for item in matched]
        snippets = []
        for item in matched:
            short_text = _shorten_summary_text(item["text"], limit=10)
            if short_text and short_text not in snippets:
                snippets.append(short_text)
            if len(snippets) >= 2:
                break

        detail_parts: list[str] = []
        for detail_text, detail_keywords in topic["details"]:
            if any(any(keyword in item["lower"] for keyword in detail_keywords) for item in matched):
                if detail_text not in detail_parts:
                    detail_parts.append(detail_text)

        if not detail_parts:
            detail_parts.append(topic["details"][0][0])

        snapshots.append(
            {
                "count": len(matched),
                "title": topic["title"],
                "range": _format_summary_range_v2(stamps),
                "details": detail_parts,
                "snippets": snippets,
            }
        )

    snapshots.sort(key=lambda item: int(item["count"]), reverse=True)
    non_generic = [item for item in snapshots if item["title"] != "Обычное общение"]
    selected = (non_generic or snapshots)[:3]
    selected = [item for item in selected if item["title"] != "Привязка и связь" or any(k in " ".join(item.get("snippets", [])) for k in ("link", "привяз", "дискорд", "телеграм"))]

    if not selected:
        source_counts: dict[str, int] = {"minecraft": 0, "telegram": 0}
        for line in lines:
            lower = line.lower()
            if "/minecraft/" in lower:
                source_counts["minecraft"] += 1
            elif "/telegram/" in lower:
                source_counts["telegram"] += 1

        parts = []
        if source_counts["minecraft"]:
            parts.append(f"Minecraft: {source_counts['minecraft']} сообщений")
        if source_counts["telegram"]:
            parts.append(f"Telegram: {source_counts['telegram']} сообщений")

        if parts:
            return (
                f"За последние {hours} часов в общих чатах почти ничего конкретного не выделилось. "
                f"По площадкам: {', '.join(parts)}."
            )

        return f"За последние {hours} часов в общих чатах почти ничего не писали."

    total = len(parsed_lines)
    intro = f"За последние {hours} часов в общих чатах было {total} сообщений. Главные мысли диалогов:"
    bullets = []
    for item in selected:
        details = "; ".join(item["details"][:2])
        snippets = item.get("snippets") or []
        snippet_text = ""
        if snippets:
            snippet_text = " Например: " + " / ".join(snippets[:2]) + "."
        bullets.append(f"• {item['range']} {item['title'].lower()}: {details}.{snippet_text}")

    return "\n".join([intro, *bullets])


def is_too_generic_summary(content: str) -> bool:
    low = normalize_text(content).lower()
    generic_phrases = (
        "общение игроков",
        "обычное игровое общение",
        "спокойный, повседневный чат",
        "без большой общей темы",
        "без ярко выраженной одной темы",
        "вопросы и короткие ответы",
        "просто общались",
        "общими чатами",
        "обычное общение",
    )
    return any(phrase in low for phrase in generic_phrases)


async def summarize_recent_chat(hours: int = 8) -> str:
    lines = get_recent_public_chat_lines(hours=hours, limit=500)
    if not lines:
        return f"За последние {hours} часов я почти ничего не увидела в общих чатах, мой хороший."

    summary_prompt = (
        "Ты — Лемонка, милая помощница сервера LemonCraft.\n"
        "Твоя задача — сжать и выделить главные темы, не упуская ничего важного.\n"
        "Нужно не просто назвать темы, а объяснить, что именно происходило, о чём спорили, что обсуждали и к чему пришли.\n"
        "Пиши конкретно, по смыслу, без воды и без общих фраз.\n"
        "Если обсуждали конфликт, укажи конфликт. Если жаловались, укажи на что. Если просили помощь, укажи какую.\n"
        "Если видишь время в сообщениях, используй его и укажи примерный промежуток.\n"
        "Не выдумывай ничего лишнего.\n"
        "Не цитируй сообщения дословно.\n"
        "Пиши по-русски, кратко, но с конкретикой.\n"
        "Формат ответа:\n"
        "1) Короткий вводный итог в 1 строку.\n"
        "2) 3-6 пунктов с главными темами.\n"
        "3) В каждом пункте: что произошло, когда это было, и чем это закончилось или к чему свелось.\n"
        "Если тема повторялась много раз, так и скажи."
        "Отвечай кратко (2–5 предложений), если пользователь не просит подробностей. Если он явно просит рассказать подробнее — дай развёрнутый ответ."
    )

    user_prompt = (
        f"Сожми и выдели главные темы, не упуская ничего, за последние {hours} часов.\n"
        f"Ниже все сообщения из общих чатов за этот период:\n\n"
        + "\n".join(lines)
    )

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=700,
            stream=False,
        )

        choice = response.choices[0]

        logger.info("Finish reason: %s", choice.finish_reason)
        logger.info("Content length: %d", len(choice.message.content or ""))
        logger.info("Content: %r", choice.message.content)

        content = ""
        try:
            choices = getattr(response, "choices", None) or []
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) or ""
        except Exception:
            content = ""

        content = normalize_text(content)
        if content and not is_too_generic_summary(content):
            return content
    except Exception as e:
        logger.warning(f"Не удалось собрать AI-сводку: {e}")

    return build_local_chat_summary(lines, hours)

def normalize_actor_key(value: Any) -> str:
    return normalize_text(value)


def get_telegram_actor_key(tg_id: int) -> str:
    linked_uuid = linked_accounts.get(tg_id)
    if linked_uuid:
        return f"uuid:{normalize_text(linked_uuid)}"
    return f"telegram:{tg_id}"


def get_minecraft_actor_key(player_uuid: str) -> str:
    player_uuid = normalize_text(player_uuid)
    return f"uuid:{player_uuid}" if player_uuid else ""


def _remember_chat_message(source: str, room: str, speaker: str, text: str, actor_key: str = "") -> None:
    source = normalize_text(source) or "unknown"
    room = normalize_text(room) or "general"
    speaker = normalize_text(speaker) or "unknown"
    actor_key = normalize_actor_key(actor_key)
    text = normalize_text(text)
    if not text:
        return

    chat_memory.append({
        "ts": time.time(),
        "source": source,
        "room": room,
        "speaker": speaker,
        "text": text,
        "actor_key": actor_key,
    })
    if len(chat_memory) > 2000:
        del chat_memory[:-2000]
    save_chat_memory()


def _candidate_memory_lines(actor_key: str = "") -> list[str]:
    if not chat_memory:
        return []

    actor_key = normalize_actor_key(actor_key)
    recent_memory = chat_memory[-500:]
    candidate_lines: list[str] = []
    for item in recent_memory:
        source = normalize_text(item.get("source"))
        room = normalize_text(item.get("room"))
        speaker = normalize_text(item.get("speaker"))
        text = normalize_text(item.get("text"))
        item_actor_key = normalize_actor_key(item.get("actor_key"))
        if not text:
            continue
        if actor_key and item_actor_key and item_actor_key != actor_key:
            continue
        candidate_lines.append(f"[{source}/{room}] {speaker}: {text}")
    return candidate_lines


def get_recent_dialog_history(actor_key: str = "", limit: int = 8) -> str:
    lines = _candidate_memory_lines(actor_key)
    if not lines:
        return ""
    return "\n".join(lines[-limit:])


def get_chat_memory_context(query: str, actor_key: str = "", limit: int = 6) -> str:
    query = normalize_text(query).lower()
    if not query or not chat_memory:
        return ""

    candidate_lines = _candidate_memory_lines(actor_key)
    if not candidate_lines and actor_key:
        candidate_lines = _candidate_memory_lines("")

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


async def observe_chat_message(source: str, room: str, speaker: str, text: str, actor_key: str = "") -> None:
    text = normalize_text(text)
    if not text:
        return
    _remember_chat_message(source, room, speaker, text, actor_key)


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


def _telegram_room_matches_summary(room: str) -> bool:
    room = normalize_text(room)
    if not room or room.startswith("private:"):
        return False

    try:
        room_id = int(room)
    except ValueError:
        return False

    room_variants = {str(room_id), str(abs(room_id))}
    if any(variant in {str(chat_id) for chat_id in SUMMARY_TELEGRAM_CHAT_IDS} for variant in room_variants):
        return True

    abs_room = str(abs(room_id))
    return any(abs_room.endswith(str(chat_id)) for chat_id in SUMMARY_TELEGRAM_CHAT_IDS)


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

ТЕБЕ ЗАПРЕЩЕНО УПОМЕНАТЬ ДРУГИЕ ПРОЕКТЫ ПО MINECRAFT, РАССКАЗЫВАТЬ О НИХ И ВЕСТИ ДИАЛОГ С ИГРОКАМИ НА ЭТУ ТЕМУ. ЗАПРЕТ НА РАЗГЛОШЕНИЕ О ДАТЕ ВАЙПА 

Тебе полностью запрещен мат. Ты осуждаешь мат и обижаешься когда человек матерится в диалоге с тобой

ТЫ НЕ БОТ

Отвечай кратко (2–5 предложений), если пользователь не просит подробностей. Если он явно просит рассказать подробнее — дай развёрнутый ответ.

РЕДКО МОЖЕШЬ ИСПОЛЬЗОВАТЬ ЭМОДЗИ
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
    load_dialogue_style()
    load_reload_notice()
    load_moderation_rules()
    load_moderation_strikes()

    PLAYERS_DATA = {}
    for tg_id, item in players.items():
        if isinstance(item, dict):
            PLAYERS_DATA[str(tg_id)] = item

    NICKNAMES_DATA = {str(tg_id): normalize_text(nick) for tg_id, nick in nicknames.items() if normalize_text(nick)}


def load_moderation_rules() -> None:
    global moderation_rules_text
    try:
        if MODERATION_RULES_FILE.exists():
            moderation_rules_text = normalize_text(MODERATION_RULES_FILE.read_text(encoding="utf-8"))
        else:
            moderation_rules_text = ""
    except Exception as e:
        moderation_rules_text = ""
        logger.warning(f"Не удалось загрузить moderation_rules.txt: {e}")


def load_moderation_strikes() -> None:
    global moderation_strikes
    try:
        moderation_strikes = load_json(MODERATION_STRIKES_FILE, {})
        if not isinstance(moderation_strikes, dict):
            moderation_strikes = {}
    except Exception as e:
        moderation_strikes = {}
        logger.warning(f"Не удалось загрузить moderation_strikes.json: {e}")


def save_moderation_strikes() -> None:
    try:
        with open(MODERATION_STRIKES_FILE, "w", encoding="utf-8") as f:
            json.dump(moderation_strikes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Не удалось сохранить moderation_strikes.json: {e}")


def get_moderation_strike_count(actor_key: str, rule_code: str) -> int:
    actor_key = normalize_actor_key(actor_key) or "global"
    rule_code = normalize_text(rule_code).lower() or "none"
    actor_bucket = moderation_strikes.setdefault(actor_key, {})
    return int(actor_bucket.get(rule_code, 0) or 0)


def increment_moderation_strike(actor_key: str, rule_code: str) -> int:
    actor_key = normalize_actor_key(actor_key) or "global"
    rule_code = normalize_text(rule_code).lower() or "none"
    actor_bucket = moderation_strikes.setdefault(actor_key, {})
    count = int(actor_bucket.get(rule_code, 0) or 0) + 1
    actor_bucket[rule_code] = count
    save_moderation_strikes()
    return count


async def ask_lemonka_ai(message_text: str, user_nick: str, actor_key: str = "") -> str:
    try:
        learn_dialogue_style(actor_key, message_text)
        relevant_info = smart_find_relevant_info(message_text)
        chat_context = get_chat_memory_context(message_text, actor_key, limit=LEMONKA_AI_CHAT_CONTEXT_LIMIT)
        dialog_history = get_recent_dialog_history(actor_key, limit=LEMONKA_AI_DIALOG_HISTORY_LIMIT)

        if relevant_info:
            return humanize_lemonka_reply(relevant_info, actor_key, message_text)

        full_prompt = LEMONKA_PROMPT
        if chat_context:
            full_prompt = f"{LEMONKA_PROMPT}\n\nНедавние наблюдения из чатов:\n{chat_context}"
        if dialog_history:
            full_prompt = f"{full_prompt}\n\nКонтекст диалога с этим игроком:\n{dialog_history}"

        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": f"{user_nick}: {message_text}"}
                ],
                temperature=0.75,
                max_tokens=LEMONKA_AI_MAX_TOKENS,
                stream=False
            ),
            timeout=LEMONKA_AI_TIMEOUT_SECONDS
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
        return humanize_lemonka_reply(content, actor_key, message_text)
    except TimeoutError:
        logger.warning("Лемонка: таймаут ответа AI, возвращаю локальный fallback")
        return lemonka_local_fallback(message_text, user_nick)
    except Exception as e:
        logger.warning(f"Локальный fallback Лемонки из-за ошибки AI: {e}")
        return lemonka_local_fallback(message_text, user_nick)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = normalize_text(text)
    if not cleaned:
        raise ValueError("empty json text")

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("json is not object")
    return parsed


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


async def send_ticket_media_to_admin(ticket_id: str, sender_name: str, sender_id: int, message: Message) -> None:
    ticket_id = normalize_text(ticket_id)
    sender_name = normalize_text(sender_name) or "Игрок"
    topic_id = ticket_topics.get(ticket_id, REPORT_TOPIC_ID)
    text = normalize_text(message.text or message.caption)
    has_media = bool(message.photo or message.video or message.document)

    if not ticket_id or not has_media:
        return

    _append_ticket_message(ticket_id, "player", text or "[медиа]")

    header = [
        "<b>Сообщение в тикет</b>",
        f"ID: <code>{html_escape(ticket_id)}</code>",
        f"От: <b>{html_escape(sender_name)}</b> [{sender_id}]",
        "",
    ]
    if text:
        header.append(html_escape(text))

    caption = "\n".join(header)

    try:
        if message.photo:
            await bot.send_photo(
                chat_id=ADMIN_GROUP_ID,
                photo=message.photo[-1].file_id,
                caption=caption,
                message_thread_id=topic_id
            )
        elif message.video:
            await bot.send_video(
                chat_id=ADMIN_GROUP_ID,
                video=message.video.file_id,
                caption=caption,
                message_thread_id=topic_id
            )
        elif message.document:
            await bot.send_document(
                chat_id=ADMIN_GROUP_ID,
                document=message.document.file_id,
                caption=caption,
                message_thread_id=topic_id
            )
    except Exception as e:
        logger.warning(f"Не удалось переслать медиа тикета {ticket_id} в Telegram: {e}")


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
    actor_key = normalize_actor_key(payload.get("actor_key") or payload.get("player_uuid") or payload.get("uuid") or "")

    if not question:
        return web.json_response({"error": "empty_question"}, status=400)

    try:
        logger.info(f"/lemonka/ask from {user_nick}: {question[:160]}")
        _remember_chat_message("minecraft", "command", user_nick, question, actor_key)
        answer = await ask_lemonka_ai(question, user_nick, actor_key)
        if answer:
            _remember_chat_message("minecraft", "command", "Лемонка", answer, actor_key)
    except Exception as e:
        logger.error(f"/lemonka/ask error: {e}")
        return web.json_response({"error": "ai_failed"}, status=500)

    return web.json_response({"answer": answer})


async def moderate_lemonka_api(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    player_name = normalize_text(payload.get("player_name") or payload.get("nick") or payload.get("username") or "Игрок")
    player_uuid = normalize_text(payload.get("player_uuid") or payload.get("uuid") or "")
    reason = normalize_text(payload.get("reason") or payload.get("text") or payload.get("message") or "")
    evidence = normalize_text(payload.get("evidence") or payload.get("details") or "")
    extra_rules = normalize_text(payload.get("rules") or payload.get("policy") or "")
    actor_key = normalize_actor_key(payload.get("actor_key") or player_uuid or player_name)

    if not player_name:
        return web.json_response({"error": "empty_player"}, status=400)

    query = " ".join(part for part in [player_name, player_uuid, reason, evidence] if part).strip()
    memory_context = get_chat_memory_context(query, "moderation", limit=8)

    rules_block = extra_rules or moderation_rules_text
    rules_tail = f"\n\nДополнительные правила модерации:\n{rules_block}" if rules_block else ""

    moderation_prompt = (
        "Ты — Лемонка, серверная помощница и модератор LemonCraft.\n"
        "Твоя задача — по описанию нарушения выбрать только одно решение: ban, mute, warn или none.\n"
        "Отвечай строго в JSON без лишнего текста.\n"
        "Формат ответа:\n"
        "{\n"
        "  \"rule_code\": \"3.1|3.1(Род.)|3.2|3.4|3.5|3.6|3.7|3.9|3.10|3.11|none\",\n"
        "  \"action\": \"ban|mute|warn|none\",\n"
        "  \"duration_minutes\": 0,\n"
        "  \"reason\": \"коротко и по делу\",\n"
        "  \"note\": \"короткая заметка для обучения\"\n"
        "}\n"
        "Правила:\n"
        "- ban/mute/warn/none выбирай по смыслу пункта правил.\n"
        "- Если нарушение подпадает под пункт с предупреждением, в rule_code укажи этот пункт.\n"
        "- Если это родня/родители в 3.1, используй rule_code = 3.1(Род.).\n"
        "- Не цитируй сообщение дословно.\n"
        "- В note пиши коротко, как для обучения: 'забанила игрока', 'замутила игрока', 'предупредила игрока'.\n"
        "- Если есть контекст прошлых случаев, учитывай его, но не копируй дословно."
        f"{rules_tail}"
    )

    user_prompt = (
        f"Игрок: {player_name}\n"
        f"UUID: {player_uuid}\n"
        f"Причина/описание: {reason or 'нет'}\n"
        f"Доказательства: {evidence or 'нет'}\n"
        f"Контекст прошлых кейсов:\n{memory_context or 'нет'}\n"
        f"Верни только JSON с решением."
    )

    try:
        response = await openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": moderation_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=220,
            stream=False,
        )

        content = ""
        try:
            choices = getattr(response, "choices", None) or []
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) or ""
        except Exception:
            content = ""

        content = normalize_text(content)
        if not content:
            raise RuntimeError("empty moderation response")

        data = extract_json_object(content)
        rule_code = normalize_text(data.get("rule_code") or "none").lower()
        action = normalize_text(data.get("action") or "none").lower()
        if action not in {"ban", "mute", "warn", "none"}:
            action = "none"
        duration_minutes = int(data.get("duration_minutes") or 0)
        if duration_minutes < 0:
            duration_minutes = 0
        decision_reason = normalize_text(data.get("reason") or reason or "без причины")
        note = normalize_text(data.get("note") or "")

        warning_rule_bases = {
            "3.2": ("mute", 30),
            "3.4": ("mute", 15),
            "3.7": ("mute", 15),
        }
        strike_count = 0
        if rule_code in warning_rule_bases:
            strike_count = increment_moderation_strike(actor_key or player_uuid or player_name, rule_code)
            if strike_count < 4:
                action = "warn"
                duration_minutes = 0
                note = note or f"предупреждение {strike_count}/3"
            else:
                action, duration_minutes = warning_rule_bases[rule_code]
                note = note or "четвёртое нарушение"

        if rule_code == "3.1(род.)":
            action = "mute"
            duration_minutes = 5 * 24 * 60
        elif rule_code == "3.1" and action == "none" and duration_minutes == 0:
            action = "mute"
            duration_minutes = 30
        elif rule_code == "3.6":
            action = "mute"
            duration_minutes = 24 * 60
        elif rule_code == "3.11":
            action = "mute"
            duration_minutes = 24 * 60

        action_labels = {
            "ban": "забанила",
            "mute": "замутила",
            "warn": "предупредила",
            "none": "не наказала",
        }
        action_label = action_labels.get(action, "разобралась с")
        timestamp_text = time.strftime("%d.%m.%Y %H:%M:%S")
        duration_text = "" if action == "warn" or action == "none" else f" на {duration_minutes} мин."
        if action == "mute" and duration_minutes >= 24 * 60:
            days = duration_minutes // (24 * 60)
            duration_text = f" на {days} дн."
        if action == "ban" and duration_minutes > 0:
            duration_text = f" на {duration_minutes} мин."

        if action != "none":
            log_text = (
                f"<b>Модерация Лемонки</b>\n"
                f"Дата/время: <code>{html_escape(timestamp_text)}</code>\n"
                f"Ник игрока: <b>{html_escape(player_name)}</b>\n"
                f"Сообщение: {html_escape(reason or evidence or 'нет')}\n"
                f"Пункт: <b>{html_escape(rule_code or 'none')}</b>\n"
                f"Решение: <b>{html_escape(action)}</b>{html_escape(duration_text)}\n"
                f"Причина: {html_escape(decision_reason)}"
            )

            _remember_chat_message(
                "minecraft",
                "moderation",
                "Лемонка",
                f"Лемонка {action_label} игрока {player_name}. {decision_reason}. {note} [rule={rule_code}, strikes={strike_count}]",
                "moderation",
            )

            try:
                await bot.send_message(
                    chat_id=ADMIN_GROUP_ID,
                    text=log_text,
                    message_thread_id=REPORT_TOPIC_ID,
                    disable_web_page_preview=True
                )
            except Exception as e:
                logger.warning(f"Не удалось отправить лог модерации в админ-беседу: {e}")

        return web.json_response({
            "rule_code": rule_code,
            "action": action,
            "duration_minutes": duration_minutes,
            "reason": decision_reason,
            "note": note,
            "strike_count": strike_count,
            "player_name": player_name,
            "player_uuid": player_uuid,
        })
    except Exception as e:
        logger.error(f"/lemonka/moderate error: {e}")
        return web.json_response({"error": "moderation_failed"}, status=500)


async def observe_chat_api(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    source = normalize_text(payload.get("source") or "unknown")
    room = normalize_text(payload.get("room") or payload.get("chat") or "general")
    speaker = normalize_text(payload.get("speaker") or payload.get("player") or payload.get("user") or "unknown")
    actor_key = normalize_actor_key(payload.get("actor_key") or payload.get("player_uuid") or payload.get("uuid") or "")
    text = normalize_text(payload.get("text") or payload.get("message") or payload.get("content"))

    if not text:
        return web.json_response({"error": "empty_text"}, status=400)

    _remember_chat_message(source, room, speaker, text, actor_key)
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
        actor_key = get_telegram_actor_key(message.from_user.id)
        await observe_chat_message(
            "telegram",
            f"private:{message.from_user.id}",
            user_nick,
            question,
            actor_key
        )
        answer = await ask_lemonka_ai(question, user_nick, actor_key)
        await observe_chat_message(
            "telegram",
            f"private:{message.from_user.id}",
            "Лемонка",
            answer,
            actor_key
        )
        await message.answer(html_escape(answer), disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка ArtemoX API: {e}")
        await message.answer("Оййй... Лемонка чуток задумалась~ ♡ Попробуй ещё разок, мой хороший? 💛")


@router.message(Command("summary", "digest", "recap"))
async def cmd_summary(message: Message, command: CommandObject):
    if message.chat.type != "private":
        await message.answer("Эта команда работает только в личке с Лемонкой.")
        return

    if not await is_admin_user(message):
        await message.answer("Эта команда только для администраторов.")
        return

    args = (command.args or "").strip()
    hours = 8
    if args:
        try:
            hours = max(1, min(24, int(args.split()[0])))
        except Exception:
            hours = 8

    status_message = await message.answer(
        f"Собираю краткую сводку за последние {hours} часов... ♡"
    )

    try:
        summary = await summarize_recent_chat(hours=hours)
        await status_message.edit_text(
            f"<b>Краткая сводка за последние {hours} часов</b>\n\n{html_escape(summary)}",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Ошибка summary: {e}")
        await status_message.edit_text(
            "Ой, не смогла быстро собрать сводку. Попробуй ещё разок, мой хороший."
        )


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
    reload_notice.clear()
    reload_notice.update({
        "requested_by": int(message.from_user.id),
        "requested_at": time.time(),
    })
    save_reload_notice()
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

    text = normalize_text(message.text or message.caption)
    speaker = message.from_user.full_name if message.from_user else "unknown"

    if message.chat.type in ("group", "supergroup"):
        if text and not text.startswith("/"):
            await observe_chat_message("telegram", str(message.chat.id), speaker, text)
        return

    if message.chat.type != "private":
        return

    has_media = bool(message.photo or message.video or message.document)
    has_text = bool(message.text and not message.text.startswith("/"))
    if not has_media and not has_text:
        return

    user_id = message.from_user.id
    if user_id in return_to_lemonka:
        return_to_lemonka.discard(user_id)
        ticket_hub_selection.pop(user_id, None)

    pending = pending_actions.get(user_id, {})
    if pending.get("mode") == "create_ticket":
        ticket_text = normalize_text(message.text or message.caption)
        if not ticket_text and not has_media:
            await message.answer("Отправь текст или медиа первого сообщения тикета.")
            return

        if not ticket_text:
            ticket_text = "[медиа]"

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
            if has_media:
                await send_ticket_media_to_admin(
                    ticket_id,
                    message.from_user.full_name,
                    user_id,
                    message,
                )
        return

    selected_ticket_id = ticket_hub_selection.get(user_id)
    if selected_ticket_id and selected_ticket_id in ticket_messages:
        ticket_text = normalize_text(message.text or message.caption)
        if ticket_text or has_media:
            if has_media:
                await send_ticket_media_to_admin(
                    selected_ticket_id,
                    message.from_user.full_name,
                    user_id,
                    message,
                )
            else:
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

    actor_key = get_telegram_actor_key(user_id)

    if text:
        await observe_chat_message("telegram", f"private:{user_id}", speaker, text, actor_key)

    if has_media:
        await forward_to_owner(message)
        return

    if not has_text:
        return

    await forward_to_owner(message)

    user_nick = nicknames.get(message.from_user.id, message.from_user.first_name)

    lemonka_reply = None
    try:
        lemonka_reply = await ask_lemonka_ai(message.text, user_nick, actor_key)
        await message.answer(html_escape(lemonka_reply), disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Ошибка ArtemoX API: {e}")
        lemonka_reply = "Оййй... Лемонка чуток задумалась~ ♡ Попробуй ещё разок, мой хороший? 💛"
        await message.answer(lemonka_reply)

    if lemonka_reply:
        await observe_chat_message(
            "telegram",
            f"private:{user_id}",
            "Лемонка",
            lemonka_reply,
            actor_key
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
    app.router.add_post("/lemonka/moderate", moderate_lemonka_api)
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


async def notify_reload_completed() -> None:
    if not reload_notice:
        return

    requested_by = int(reload_notice.get("requested_by") or OWNER_ID)
    try:
        await bot.send_message(
            chat_id=requested_by,
            text="Лемонка успешно перезагрузилась ♡"
        )
    except Exception as e:
        logger.warning(f"Не удалось отправить уведомление о перезагрузке: {e}")
    finally:
        clear_reload_notice()


async def main():
    logger.info("Бот запущен")
    load_ai_data()
    load_ticket_state()
    load_link_state()
    await load_or_upload_kits()
    await start_web()
    await notify_reload_completed()
    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        await stop_web()


if __name__ == "__main__":
    asyncio.run(main())
