# handlers.py
import logging
import math
import traceback
from datetime import datetime, timedelta
from typing import Tuple, List

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


# --- Вспомогательные функции для обработки ответов ---
def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    """Извлекает ссылки и создаёт красивые кнопки"""
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)

    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "")
            clean_url = clean_url.strip('.,;:!?()"\'[]{}')
            if not clean_url:
                continue

            label = "🔗 Открыть ссылку"
            if "roadmap" in clean_url.lower():
                label = "🗺 Дорожная карта"
            elif "Business-card" in clean_url or "avick23.github.io" in clean_url:
                label = "👤 О преподавателе"
            elif "t.me" in clean_url:
                label = "💬 Telegram"
            elif "calendar" in clean_url.lower():
                label = "📅 Выбрать время"

            buttons.append([InlineKeyboardButton(label, url=clean_url)])

        clean_text = re.sub(url_pattern, '', text).strip()
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        clean_text = re.sub(r'\(\s*\)', '', clean_text).strip()
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


async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "🗺 <b>Дорожные карты обучения</b>\n\nВыберите направление:"
    await update.message.reply_text(
        text,
        reply_markup=AppleKeyboards.roadmaps_menu(),
        parse_mode="HTML"
    )


# --- Обработчик callback-запросов ---
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
            "🗓 <b>Запись на консультацию</b>\n\nВыберите удобный способ:",
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

    # --- Обработка стандартных вопросов меню ---
    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {
            "menu_cost": "стоимость",
            "menu_method": "метод выстраданного познания",
            "menu_about": "кто такой алексей"
        }
        kb_index = context.bot_data.get('kb_index')
        if not kb_index:
            await query.edit_message_text(
                "⚠️ База знаний недоступна",
                reply_markup=AppleKeyboards.back_button()
            )
            return

        answer, score, candidates = search_knowledge_base(q_map[data], kb_index)
        if not answer:
            await query.edit_message_text(
                AppleStyleMessages.NOT_FOUND,
                reply_markup=AppleKeyboards.back_button(),
                parse_mode="HTML"
            )
            return

        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)

        ans_idx = candidates[0]['index'] if candidates else 0
        save_question_for_answer(context, user_id, ans_idx, q_map[data])

        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])

        url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))

        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    # --- История ---
    if data == "menu_history":
        await show_history(update, context, 0)
        return

    if data.startswith("history_page_"):
        page = int(data.split("_")[2])
        await show_history(update, context, page)
        return

    # --- Отзыв о боте ---
    if data == "menu_feedback":
        await query.edit_message_text(
            AppleStyleMessages.FEEDBACK_PROMPT,
            reply_markup=AppleKeyboards.back_button()
        )
        # Переводим бота в режим ожидания отзыва
        context.user_data['awaiting_feedback'] = True
        return

    # --- FAQ ---
    if data == "menu_faq":
        kb_index = context.bot_data.get('kb_index')
        if not kb_index:
            await query.edit_message_text("⚠️ База знаний недоступна", reply_markup=AppleKeyboards.back_button())
            return
        # Берём первые 5 тем (индексы с высокими keywords)
        faq_indices = list(range(min(5, len(kb_index.items))))
        faq_items = [(kb_index.items[i]['original_keywords'][0] if kb_index.items[i]['original_keywords'] else f"Тема {i+1}", i) for i in faq_indices]
        await query.edit_message_text(
            AppleStyleMessages.FAQ_TITLE,
            reply_markup=AppleKeyboards.faq_menu(faq_items),
            parse_mode="HTML"
        )
        return

    if data.startswith("faq_"):
        idx = int(data.split("_")[1])
        kb_index = context.bot_data.get('kb_index')
        if not kb_index or not kb_index.is_valid_index(idx):
            await query.answer("Тема не найдена", show_alert=True)
            return
        answer = kb_index.items[idx]["context"]
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))
        save_question_for_answer(context, user_id, idx, "FAQ")
        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    # --- Уточнение вопроса ---
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
            await query.answer("Ответ не найден", show_alert=True)
            return

        answer = kb_index.items[idx]["context"]
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)

        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])

        save_question_for_answer(context, user_id, idx, "Уточняющий вопрос")
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))

        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    # --- Консультация ---
    if data == "consultation":
        await consultation_callback(update, context)
        return

    # --- Обратная связь на ответ ---
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
            reply_markup=AppleKeyboards.admin_menu(),
            parse_mode="HTML"
        )
        return

    if data.startswith("admin_consult_"):
        page = int(data.split("_")[2])
        await admin_show_list(update, context, "consult", page)
        return

    if data.startswith("admin_like_"):
        page = int(data.split("_")[2])
        await admin_show_list(update, context, "like", page)
        return

    if data.startswith("admin_dislike_"):
        page = int(data.split("_")[2])
        await admin_show_list(update, context, "dislike", page)
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
        page = int(query.message.reply_markup.inline_keyboard[-1][-1].callback_data.split("_")[-1])
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
        # Формат: admin_add_unknown_<index>_<page>? или просто индекс
        parts = data.split("_")
        idx = int(parts[3])
        await admin_add_answer_prompt(update, context, idx)
        return


# --- Функция показа истории ---
async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE, page: int):
    query = update.callback_query
    user_id = update.effective_user.id
    ctx = get_user_context(context, user_id)
    history_q = list(ctx.get("history", []))
    history_a = list(ctx.get("answers", []))
    if not history_q:
        await query.edit_message_text(
            AppleStyleMessages.HISTORY_EMPTY,
            reply_markup=AppleKeyboards.back_button(),
            parse_mode="HTML"
        )
        return

    total = len(history_q)
    total_pages = math.ceil(total / ITEMS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE

    text = AppleStyleMessages.HISTORY_TITLE.format(count=total) + "\n\n"
    for i in range(start, min(end, total)):
        q = history_q[i]
        a = history_a[i] if i < len(history_a) else "…"
        # Обрезаем длинные ответы
        a_short = a[:100] + "…" if len(a) > 100 else a
        text += f"<b>❓ {i+1}. {q}</b>\n💬 {a_short}\n\n"

    markup = AppleKeyboards.history_menu(list(zip(history_q, history_a)), page, total_pages)
    await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")


# --- Обработка отзыва о боте ---
async def handle_feedback_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if context.user_data.get('awaiting_feedback'):
        feedback_text = update.message.text
        # Пересылаем админу
        await context.bot.send_message(
            ADMIN_USER_ID,
            f"💬 <b>Отзыв от пользователя</b>\n"
            f"👤 {update.effective_user.first_name} (@{update.effective_user.username})\n"
            f"🆔 {user_id}\n\n{feedback_text}",
            parse_mode="HTML"
        )
        await update.message.reply_text(
            AppleStyleMessages.FEEDBACK_SENT,
            reply_markup=AppleKeyboards.back_button(),
            parse_mode="HTML"
        )
        context.user_data['awaiting_feedback'] = False
        return True
    return False


# --- Консультация ---
async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    consultations = load_json(CONSULTATIONS_FILE)
    recent = [c for c in consultations
              if c.get("user_id") == user.id and
              datetime.now() - datetime.strptime(c.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)]
    if recent:
        await query.edit_message_text(
            "✅ <b>Вы уже записаны</b>\n\nВаша заявка обрабатывается. Ожидайте связи!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=CALENDAR_URL)]]),
            parse_mode="HTML"
        )
        return

    consultations.append({
        "user_id": user.id,
        "username": user.username or "Нет",
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "timestamp": timestamp
    })
    save_json(CONSULTATIONS_FILE, consultations)

    try:
        await context.bot.send_message(
            ADMIN_USER_ID,
            f"🔔 <b>Новая заявка!</b>\n\n"
            f"👤 {user.first_name or 'Без имени'}\n"
            f"📱 @{user.username or 'нет username'}\n"
            f"🆔 {user.id}",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Failed to notify admin: {e}")

    keyboard = [[InlineKeyboardButton("📅 Выбрать время в календаре", url=CALENDAR_URL)]]
    await query.edit_message_text(
        AppleStyleMessages.CONSULTATION_SUCCESS,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


# --- Обратная связь на ответ ---
async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user = query.from_user
    await query.answer()

    fb_type = "like" if data.startswith("like_") else "dislike"
    try:
        idx = int(data.split("_")[1])
    except (IndexError, ValueError):
        logger.error(f"Invalid callback data: {data}")
        await query.answer("Ошибка данных", show_alert=True)
        return

    kb_index = context.bot_data.get('kb_index')
    if not kb_index or not kb_index.is_valid_index(idx):
        await query.answer("Ответ не найден", show_alert=True)
        return

    answer = kb_index.items[idx]["context"]
    question = get_question_for_answer(context, user.id, idx)
    if question == "???":
        ctx = get_user_context(context, user.id)
        history = ctx.get("history", [])
        if history:
            question = list(history)[-1]

    feedback_list = load_json(FEEDBACK_FILE)
    feedback_list.append({
        "type": fb_type,
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "user_id": user.id,
        "username": user.username,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(FEEDBACK_FILE, feedback_list)

    if fb_type == "like":
        new_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("💚 Спасибо за оценку!", callback_data="ignore")]])
        await query.edit_message_reply_markup(new_keyboard)
    else:
        new_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]])
        await query.edit_message_reply_markup(new_keyboard)
        await query.message.reply_text(AppleStyleMessages.FEEDBACK_DISLIKE, parse_mode="HTML")
        try:
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"👎 <b>Дизлайк</b>\n\n"
                f"❓ <b>Вопрос:</b> {question}\n"
                f"💬 <b>Ответ:</b> {answer[:100]}...\n"
                f"👤 @{user.username or user.id}",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")


# --- Основной обработчик сообщений ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    # Проверяем, не ждём ли мы отзыв
    if await handle_feedback_message(update, context):
        return

    user_id = update.effective_user.id
    user_question = update.message.text.strip()

    cleanup_inactive_users(context)
    get_user_context(context, user_id)
    update_user_activity(context, user_id)

    search_query = get_contextual_question(context, user_id, user_question)
    kb_index = context.bot_data.get('kb_index')
    if not kb_index:
        await update.message.reply_text("⚠️ База знаний временно недоступна. Попробуйте позже.")
        return

    answer, score, candidates = search_knowledge_base(search_query, kb_index)
    final_answer = None

    if score > 3.5 and answer:
        final_answer = answer
    elif score > 1.5 and candidates:
        keyboard = [
            [InlineKeyboardButton(f"💬 {c['topic']}", callback_data=f"clarify_{c['index']}")]
            for c in candidates
        ]
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        await update.message.reply_text(
            AppleStyleMessages.CLARIFY_PROMPT,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return
    else:
        # Попытка нечёткого поиска
        suggestion = get_fuzzy_suggestion(user_question, kb_index)
        if suggestion:
            answer, score, candidates = search_knowledge_base(suggestion, kb_index)
            if score > 1.5:
                final_answer = answer
            if score < 3.5 and candidates:
                keyboard = [
                    [InlineKeyboardButton(f"💡 {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]
                ]
                await update.message.reply_text(
                    AppleStyleMessages.FUZZY_SUGGESTION,
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

        await update.message.reply_text(
            AppleStyleMessages.NOT_FOUND,
            reply_markup=AppleKeyboards.main_menu(is_returning=True, is_admin=(user_id == ADMIN_USER_ID)),
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

    await update.message.reply_text(
        display_text,
        reply_markup=InlineKeyboardMarkup(url_buttons),
        disable_web_page_preview=True,
        parse_mode="HTML"
    )


# --- Админские функции ---
async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str, page: int = 0):
    query = update.callback_query
    await query.answer()

    items = []
    title = ""
    empty_msg = ""
    if data_type == "consult":
        items = load_json(CONSULTATIONS_FILE)
        title = "📋 Заявки на консультацию"
        empty_msg = "Заявок пока нет."
    elif data_type == "like":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "like"]
        title = "💚 Лайки"
        empty_msg = "Лайков пока нет."
    elif data_type == "dislike":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "dislike"]
        title = "👎 Дизлайки"
        empty_msg = "Жалоб пока нет."
    elif data_type == "unknown":
        items = load_json(UNKNOWN_FILE)
        title = "❓ Неизвестные вопросы"
        empty_msg = "Неизвестных вопросов нет."

    total = len(items)
    total_pages = math.ceil(total / ITEMS_PER_PAGE) if total else 1
    page = max(0, min(page, total_pages - 1))

    text = f"<b>{title}</b>\nВсего: {total}\n\n"
    if not items:
        text += f"<i>{empty_msg}</i>"
    else:
        start = page * ITEMS_PER_PAGE
        end = start + ITEMS_PER_PAGE
        for i, item in enumerate(items[start:end], start=start):
            if data_type == "consult":
                text += f"{i+1}. {item.get('first_name', '')} @{item.get('username', '')}\n   ⏰ {item.get('timestamp', '')}\n\n"
            elif data_type == "unknown":
                q = item.get('question', '???')
                text += f"{i+1}. {q[:100]}{'…' if len(q) > 100 else ''}\n"
                # Кнопка "Добавить ответ" будет добавлена после списка
            else:
                q = item.get('question', '???')
                text += f"{i+1}. {q[:100]}{'…' if len(q) > 100 else ''}\n"

    keyboard = []
    # Пагинация
    if total_pages > 1:
        nav_row = AppleKeyboards.pagination(f"admin_{data_type}", page, total_pages)
        keyboard.append(nav_row)

    # Кнопки действий
    if items:
        if data_type == "unknown":
            # Для неизвестных вопросов добавим кнопку для каждого элемента (в следующей версии можно сделать отдельный просмотр)
            # Здесь мы добавим общую кнопку "Добавить ответ" для первого вопроса на странице (упрощённо)
            first_idx = start
            keyboard.append([InlineKeyboardButton("➕ Добавить ответ на первый вопрос", callback_data=f"admin_add_unknown_{first_idx}")])
        keyboard.append([InlineKeyboardButton("🗑 Очистить всё", callback_data=f"admin_clear_{data_type}")])

    keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="admin_menu")])

    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")


async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Статистика по пользователям
    user_contexts = context.bot_data.get('user_contexts', {})
    total_users = len(user_contexts)
    now = datetime.now()
    active_day = sum(1 for ctx in user_contexts.values() if now - ctx.get("last_activity", now) < timedelta(hours=24))
    active_week = sum(1 for ctx in user_contexts.values() if now - ctx.get("last_activity", now) < timedelta(days=7))

    # Всего вопросов в истории
    total_questions = sum(len(ctx.get("history", [])) for ctx in user_contexts.values())

    text = AppleStyleMessages.STATS_TITLE.format(
        total_users=total_users,
        active_day=active_day,
        active_week=active_week,
        total_questions=total_questions
    )

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data="admin_menu")]]
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")


async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, item_type: str):
    query = update.callback_query
    await query.answer()

    if item_type == "consult":
        save_json(CONSULTATIONS_FILE, [])
    elif item_type in ["like", "dislike"]:
        fb = load_json(FEEDBACK_FILE)
        save_json(FEEDBACK_FILE, [x for x in fb if x.get("type") != item_type])
    elif item_type == "unknown":
        save_json(UNKNOWN_FILE, [])

    await query.edit_message_text("✅ <b>Очищено успешно</b>", parse_mode="HTML")
    # Возвращаемся в админ-меню
    await query.message.reply_text(
        AppleStyleMessages.ADMIN_PANEL_TITLE,
        reply_markup=AppleKeyboards.admin_menu(),
        parse_mode="HTML"
    )


async def admin_add_answer_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE, item_index: int):
    """Запрашивает у админа ответ на выбранный неизвестный вопрос."""
    query = update.callback_query
    await query.answer()

    unknown_list = load_json(UNKNOWN_FILE)
    if item_index >= len(unknown_list):
        await query.edit_message_text("❌ Вопрос не найден.", reply_markup=AppleKeyboards.back_button("admin_unknown_0"))
        return

    question = unknown_list[item_index]["question"]
    # Сохраняем индекс в user_data для следующего шага
    context.user_data['adding_answer_for'] = item_index
    await query.edit_message_text(
        AppleStyleMessages.ADD_ANSWER_PROMPT.format(question=question),
        parse_mode="HTML"
    )
    # Переводим бота в режим ожидания ответа
    context.user_data['awaiting_answer'] = True


async def handle_add_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обрабатывает ввод ответа админом для добавления в базу знаний."""
    if not context.user_data.get('awaiting_answer'):
        return False

    user_id = update.effective_user.id
    if user_id != ADMIN_USER_ID:
        return False

    answer_text = update.message.text.strip()
    item_index = context.user_data.get('adding_answer_for')
    if item_index is None:
        return False

    unknown_list = load_json(UNKNOWN_FILE)
    if item_index >= len(unknown_list):
        await update.message.reply_text("❌ Ошибка: вопрос не найден.")
        context.user_data['awaiting_answer'] = False
        return True

    question = unknown_list[item_index]["question"]

    # Загружаем текущую базу знаний
    kb_data = load_json(MAIN_JSON)
    # Создаём новую запись
    new_entry = {
        "context": answer_text,
        "keywords": [question]  # используем вопрос как ключевое слово
    }
    kb_data.append(new_entry)
    save_json(MAIN_JSON, kb_data)

    # Перестраиваем индекс
    kb_index = preprocess_knowledge_base(kb_data)
    context.bot_data['kb_index'] = kb_index

    # Удаляем вопрос из неизвестных
    unknown_list.pop(item_index)
    save_json(UNKNOWN_FILE, unknown_list)

    await update.message.reply_text(
        AppleStyleMessages.ANSWER_ADDED,
        reply_markup=AppleKeyboards.back_button("admin_unknown_0"),
        parse_mode="HTML"
    )

    context.user_data['awaiting_answer'] = False
    context.user_data.pop('adding_answer_for', None)
    return True


# --- Глобальный обработчик ошибок ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Что-то пошло не так.\n\nПопробуйте позже или напишите /start",
                parse_mode="HTML"
            )
        except Exception:
            pass
    if ADMIN_USER_ID:
        try:
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"❌ <b>ERROR:</b>\n<pre>{tb_string[:4000]}</pre>",
                parse_mode="HTML"
            )
        except Exception:
            pass