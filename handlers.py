# handlers.py
import logging
import math
import traceback
from datetime import datetime, timedelta
from typing import Tuple, List
import re

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import (
    ADMIN_USER_ID, CONSULTATIONS_FILE, UNKNOWN_FILE, FEEDBACK_FILE,
    MAIN_JSON, ITEMS_PER_PAGE, CALENDAR_URL
)
from messages import AppleStyleMessages, AppleKeyboards
from utils import (
    load_json, save_json, search_knowledge_base, get_fuzzy_suggestion,
    get_user_context, update_user_activity, cleanup_inactive_users,
    save_question_for_answer, get_question_for_answer, get_contextual_question,
    add_to_history, preprocess_knowledge_base, KBIndex
)

logger = logging.getLogger(__name__)


# --- Вспомогательные функции ---
def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)

    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "").strip('.,;:!?()"\'[]{}')
            if not clean_url: continue

            label = "🔗 Ссылка"
            if "roadmap" in clean_url.lower(): label = "🗺 Карта"
            elif "Business-card" in clean_url: label = "👤 Преподаватель"
            elif "calendar" in clean_url.lower(): label = "📅 Календарь"

            buttons.append([InlineKeyboardButton(label, url=clean_url)])

        clean_text = re.sub(url_pattern, '', text).strip()
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        return clean_text, buttons

    return text, []


# --- Команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    cleanup_inactive_users(context)
    get_user_context(context, user_id)
    update_user_activity(context, user_id)

    is_returning = user_id in context.bot_data.get('user_contexts', {})
    is_admin = (user_id == ADMIN_USER_ID)

    text = AppleStyleMessages.WELCOME_RETURNING if is_returning else AppleStyleMessages.WELCOME
    await update.message.reply_text(
        text,
        reply_markup=AppleKeyboards.main_menu(is_returning, is_admin),
        parse_mode="HTML"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(AppleStyleMessages.HELP, parse_mode="HTML")


# --- Callback Handler ---
async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    await query.answer()

    user_id = update.effective_user.id
    update_user_activity(context, user_id)

    # --- Главное меню ---
    if data == "menu_main":
        is_admin = (user_id == ADMIN_USER_ID)
        await query.edit_message_text(
            AppleStyleMessages.WELCOME_RETURNING,
            reply_markup=AppleKeyboards.main_menu(is_returning=True, is_admin=is_admin),
            parse_mode="HTML"
        )
        return

    if data == "menu_consult":
        await query.edit_message_text(
            "🗓 <b>Запись на консультацию</b>",
            reply_markup=AppleKeyboards.consult_menu(),
            parse_mode="HTML"
        )
        return

    if data == "menu_roadmaps":
        await query.edit_message_text(
            "🗺 <b>Дорожные карты</b>",
            reply_markup=AppleKeyboards.roadmaps_menu(),
            parse_mode="HTML"
        )
        return

    # --- Стандартные вопросы (Cost, Method, About) ---
    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {
            "menu_cost": "стоимость",
            "menu_method": "метод выстраданного познания",
            "menu_about": "кто такой алексей"
        }
        kb_index = context.bot_data.get('kb_index')
        if not kb_index:
            await query.edit_message_text("⚠️ Ошибка базы данных", reply_markup=AppleKeyboards.back_button())
            return

        answer, score, candidates = search_knowledge_base(q_map[data], kb_index)
        if not answer:
            await query.edit_message_text(AppleStyleMessages.NOT_FOUND, parse_mode="HTML")
            return

        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)

        ans_idx = candidates[0]['index'] if candidates else 0
        save_question_for_answer(context, user_id, ans_idx, q_map[data])

        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        
        url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))

        await query.edit_message_text(
            display_text, reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True, parse_mode="HTML"
        )
        return

    # --- Уточнение вопроса (когда пользователь нажал "Не то") ---
    if data.startswith("clarify_"):
        if data == "clarify_none":
            await query.edit_message_text(
                "Хорошо, попробуйте сформулировать иначе.",
                reply_markup=AppleKeyboards.back_button()
            )
            return

        idx = int(data.split("_")[1])
        kb_index = context.bot_data.get('kb_index')
        if not kb_index or not kb_index.is_valid_index(idx):
            await query.answer("Ошибка", show_alert=True)
            return

        answer = kb_index.items[idx]["context"]
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)

        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])

        save_question_for_answer(context, user_id, idx, "Уточнение")
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))

        await query.edit_message_text(
            display_text, reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True, parse_mode="HTML"
        )
        return

    # --- Консультация ---
    if data == "consultation":
        await consultation_callback(update, context)
        return

    # --- Обратная связь ---
    if data.startswith("like_") or data.startswith("dislike_"):
        await feedback_callback(update, context)
        return

    # --- Админ-панель ---
    if user_id != ADMIN_USER_ID:
        await query.answer("Доступ запрещён", show_alert=True)
        return

    if data == "admin_menu":
        await query.edit_message_text(
            AppleStyleMessages.ADMIN_PANEL_TITLE,
            reply_markup=AppleKeyboards.admin_menu(), parse_mode="HTML"
        )
        return

    if data.startswith("admin_consult_"):
        page = int(data.split("_")[2])
        await admin_show_list(update, context, "consult", page)
        return

    if data.startswith("admin_unknown_"):
        page = int(data.split("_")[2])
        await admin_show_list(update, context, "unknown", page)
        return
        
    if data == "admin_stats":
        await admin_stats(update, context)
        return

    if data.startswith("admin_clear_"):
        item_type = data.replace("admin_clear_", "")
        page = 0 # default fallback
        await query.edit_message_text(
            f"⚠️ Очистить все {item_type}?",
            reply_markup=AppleKeyboards.confirm_clear(item_type, page),
            parse_mode="HTML"
        )
        return

    if data.startswith("admin_do_clear_"):
        item_type = data.replace("admin_do_clear_", "")
        await admin_do_clear(update, context, item_type)
        return

    if data.startswith("admin_add_unknown_"):
        parts = data.split("_")
        idx = int(parts[3])
        await admin_add_answer_prompt(update, context, idx)
        return


# --- Основная логика сообщений (Apple Style Flow) ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_question = update.message.text.strip()

    cleanup_inactive_users(context)
    get_user_context(context, user_id)
    update_user_activity(context, user_id)

    search_query = get_contextual_question(context, user_id, user_question)
    kb_index = context.bot_data.get('kb_index')
    if not kb_index:
        await update.message.reply_text("⚠️ Ошибка базы данных.")
        return

    answer, score, candidates = search_knowledge_base(search_query, kb_index)
    final_answer = None
    best_candidate_idx = candidates[0]['index'] if candidates else 0

    # APPLE LOGIC: Предполагай, а не спрашивай.
    # Если уверенность высокая (> 3.5) -> Даем ответ.
    # Если уверенность средняя (> 1.0) -> Даем ответ НО добавляем кнопку "Уточнить".
    # Если низкая -> "Не знаю".

    if score > 3.5 and answer:
        final_answer = answer
    
    elif score > 1.0 and candidates:
        # Мы достаточно уверены, чтобы дать ответ сразу,
        # но сохраняем кандидаты, чтобы пользователь мог выбрать другой, если это не то.
        final_answer = answer
        # Сохраняем кандидатов для возможного уточнения
        context.user_data['last_candidates'] = candidates

    else:
        # Попытка нечеткого поиска
        suggestion = get_fuzzy_suggestion(user_question, kb_index)
        if suggestion:
            answer, score, candidates = search_knowledge_base(suggestion, kb_index)
            if score > 1.0:
                final_answer = answer
                context.user_data['last_candidates'] = candidates
            else:
                # Предлагаем вариант одной кнопкой
                keyboard = [[InlineKeyboardButton(f"💡 Вы имели в виду: {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]]
                await update.message.reply_text(
                    AppleStyleMessages.CLARIFY_PROMPT,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode="HTML"
                )
                return

    if not final_answer:
        # Неизвестный вопрос
        unknown = load_json(UNKNOWN_FILE)
        unknown.append({
            "question": user_question,
            "user_id": user_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_json(UNKNOWN_FILE, unknown)

        # Сразу предлагаем консультацию
        keyboard = [[InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="consultation")]]
        await update.message.reply_text(
            AppleStyleMessages.NOT_FOUND,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return

    # Формирование ответа
    clean_answer = final_answer.replace("[add_button]", "").strip()
    display_text, url_buttons = extract_links_and_buttons(clean_answer)

    ans_idx = 0
    if candidates and candidates[0]['context'] == final_answer:
        ans_idx = candidates[0]['index']
    else:
        for i, item in enumerate(kb_index.items):
            if item['context'] == final_answer:
                ans_idx = i
                break

    save_question_for_answer(context, user_id, ans_idx, user_question)
    add_to_history(context, user_id, user_question, display_text)

    if "[add_button]" in final_answer:
        url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])

    url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))
    
    # Если уверенность была средней, добавляем кнопку "Другие варианты"
    if score <= 3.5 and context.user_data.get('last_candidates'):
        url_buttons.append([InlineKeyboardButton("❓ Это не то, показать другие варианты", callback_data=f"clarify_{ans_idx}_show_all")])

    await update.message.reply_text(
        display_text,
        reply_markup=InlineKeyboardMarkup(url_buttons),
        disable_web_page_preview=True,
        parse_mode="HTML"
    )


# --- Вспомогательные функции админы и прочее (без изменений логики, только очистка) ---

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    consultations = load_json(CONSULTATIONS_FILE)
    recent = [c for c in consultations if c.get("user_id") == user.id and 
              datetime.now() - datetime.strptime(c.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)]
    
    if recent:
        await query.edit_message_text(
            "✅ <b>Вы уже записаны</b>\n\nОжидайте связи!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=CALENDAR_URL)]]),
            parse_mode="HTML"
        )
        return

    consultations.append({
        "user_id": user.id, "username": user.username or "Нет",
        "first_name": user.first_name or "", "timestamp": timestamp
    })
    save_json(CONSULTATIONS_FILE, consultations)

    try:
        await context.bot.send_message(
            ADMIN_USER_ID,
            f"🔔 <b>Новая заявка!</b>\n👤 {user.first_name}\n📱 @{user.username or 'нет'}\n🆔 {user.id}",
            parse_mode="HTML"
        )
    except Exception: pass

    keyboard = [[InlineKeyboardButton("📅 Выбрать время в календаре", url=CALENDAR_URL)]]
    await query.edit_message_text(
        AppleStyleMessages.CONSULTATION_SUCCESS,
        reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML"
    )

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user = query.from_user
    await query.answer()

    fb_type = "like" if data.startswith("like_") else "dislike"
    try:
        idx = int(data.split("_")[1])
    except:
        await query.answer("Ошибка", show_alert=True)
        return

    kb_index = context.bot_data.get('kb_index')
    if not kb_index or not kb_index.is_valid_index(idx): return

    answer = kb_index.items[idx]["context"]
    question = get_question_for_answer(context, user.id, idx)
    
    feedback_list = load_json(FEEDBACK_FILE)
    feedback_list.append({
        "type": fb_type, "question": question, "answer": answer[:200],
        "user_id": user.id, "username": user.username,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(FEEDBACK_FILE, feedback_list)

    if fb_type == "like":
        await query.edit_message_reply_markup(InlineKeyboardMarkup([[InlineKeyboardButton("💚 Спасибо!", callback_data="ignore")]]))
    else:
        await query.edit_message_reply_markup(InlineKeyboardMarkup([[InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]]))
        try:
            await context.bot.send_message(ADMIN_USER_ID, f"👎 Дизлайк\nВопрос: {question}\nОтвет: {answer[:100]}", parse_mode="HTML")
        except: pass

# --- Админские функции (сжатые) ---
async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str, page: int = 0):
    query = update.callback_query
    await query.answer()
    items = []
    if data_type == "consult":
        items = load_json(CONSULTATIONS_FILE)
        title = "📋 Заявки"
    elif data_type == "unknown":
        items = load_json(UNKNOWN_FILE)
        title = "❓ Неизвестные"
    else: return

    total = len(items)
    total_pages = math.ceil(total / ITEMS_PER_PAGE) if total else 1
    page = max(0, min(page, total_pages - 1))

    text = f"<b>{title}</b> (Всего: {total})\n\n"
    start = page * ITEMS_PER_PAGE
    
    if not items:
        text += "<i>Пусто</i>"
    else:
        for i, item in enumerate(items[start:start+ITEMS_PER_PAGE], start=start):
            if data_type == "consult":
                text += f"{i+1}. {item.get('first_name', '')} @{item.get('username', '')}\n"
            else:
                q = item.get('question', '???')
                text += f"{i+1}. {q[:80]}...\n"

    keyboard = []
    if total_pages > 1:
        keyboard.append(AppleKeyboards.pagination(f"admin_{data_type}", page, total_pages))
    
    if data_type == "unknown" and items:
        keyboard.append([InlineKeyboardButton("➕ Добавить ответ на первый", callback_data=f"admin_add_unknown_{start}")])
        
    keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="admin_menu")])
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_contexts = context.bot_data.get('user_contexts', {})
    total_users = len(user_contexts)
    now = datetime.now()
    active_day = sum(1 for ctx in user_contexts.values() if now - ctx.get("last_activity", now) < timedelta(hours=24))
    total_questions = sum(len(ctx.get("history", [])) for ctx in user_contexts.values())
    
    text = AppleStyleMessages.STATS_TITLE.format(total_users=total_users, active_day=active_day, active_week="N/A", total_questions=total_questions)
    await query.edit_message_text(text, reply_markup=AppleKeyboards.back_button("admin_menu"), parse_mode="HTML")

async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, item_type: str):
    query = update.callback_query
    await query.answer()
    if item_type == "consult": save_json(CONSULTATIONS_FILE, [])
    elif item_type == "unknown": save_json(UNKNOWN_FILE, [])
    await query.edit_message_text("✅ Очищено", reply_markup=AppleKeyboards.back_button("admin_menu"), parse_mode="HTML")

async def admin_add_answer_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE, item_index: int):
    query = update.callback_query
    await query.answer()
    unknown_list = load_json(UNKNOWN_FILE)
    if item_index >= len(unknown_list): return
    question = unknown_list[item_index]["question"]
    context.user_data['adding_answer_for'] = item_index
    await query.edit_message_text(AppleStyleMessages.ADD_ANSWER_PROMPT.format(question=question), parse_mode="HTML")
    context.user_data['awaiting_answer'] = True

async def handle_add_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not context.user_data.get('awaiting_answer'): return False
    user_id = update.effective_user.id
    if user_id != ADMIN_USER_ID: return False

    answer_text = update.message.text.strip()
    item_index = context.user_data.get('adding_answer_for')
    unknown_list = load_json(UNKNOWN_FILE)
    if item_index is None or item_index >= len(unknown_list): return False

    question = unknown_list[item_index]["question"]
    kb_data = load_json(MAIN_JSON)
    kb_data.append({"context": answer_text, "keywords": [question]})
    save_json(MAIN_JSON, kb_data)

    kb_index = preprocess_knowledge_base(kb_data)
    context.bot_data['kb_index'] = kb_index

    unknown_list.pop(item_index)
    save_json(UNKNOWN_FILE, unknown_list)

    await update.message.reply_text(AppleStyleMessages.ANSWER_ADDED, parse_mode="HTML")
    context.user_data['awaiting_answer'] = False
    return True

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception:", exc_info=context.error)
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text("⚠️ Произошла ошибка. Попробуйте позже.")
        except: pass