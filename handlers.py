import math
from datetime import datetime, timedelta
from typing import List, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import (
    ADMIN_USER_ID, CONSULTATIONS_FILE, UNKNOWN_FILE, FEEDBACK_FILE,
    CALENDAR_URL, ROADMAPS, ITEMS_PER_PAGE,
    SCORE_DIRECT_ANSWER, SCORE_CLARIFY, logger
)
from utils import (
    kb_index,
    user_contexts,                     # добавить
    load_json, save_json,
    search_knowledge_base, get_user_context, update_user_activity,
    save_question_for_answer, get_question_for_answer,
    add_favorite, remove_favorite, get_favorites,
    extract_links_and_buttons, cleanup_inactive_users,
    get_contextual_question             # добавить
)

try:
    from thefuzz import process
    FUZZY_ENABLED = True
except ImportError:
    FUZZY_ENABLED = False
    print("⚠️ Библиотека thefuzz не установлена. Поиск опечаток отключен.")


# ====================== Клавиатуры ======================

class AppleKeyboards:
    @staticmethod
    def main_menu(is_returning: bool = False) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="menu_consult")],
            [
                InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"),
                InlineKeyboardButton("🗺 Карты обучения", callback_data="menu_roadmaps")
            ],
            [
                InlineKeyboardButton("🧠 О методе", callback_data="menu_method"),
                InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")
            ],
            [InlineKeyboardButton("⭐ Избранное", callback_data="menu_favorites")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def feedback_buttons(answer_index: int) -> List[List[InlineKeyboardButton]]:
        return [
            [
                InlineKeyboardButton("👍 Полезно", callback_data=f"like_{answer_index}"),
                InlineKeyboardButton("👎 Не помогло", callback_data=f"dislike_{answer_index}"),
                InlineKeyboardButton("⭐ В избранное", callback_data=f"fav_add_{answer_index}")
            ]
        ]

    @staticmethod
    def consult_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("📅 Выбрать время в календаре", url=CALENDAR_URL)],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def roadmaps_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🐍 Python", url=ROADMAPS["python"])],
            [InlineKeyboardButton("⚡ Backend", url=ROADMAPS["backend"])],
            [InlineKeyboardButton("🐹 Golang", url=ROADMAPS["golang"])],
            [InlineKeyboardButton("🔧 DevOps", url=ROADMAPS["devops"])],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def back_button(callback_data: str = "menu_main") -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=callback_data)]])

    @staticmethod
    def favorites_menu(favorite_indices: List[int]) -> InlineKeyboardMarkup:
        if not favorite_indices:
            return AppleKeyboards.back_button()

        keyboard = []
        for idx in favorite_indices[:5]:
            if kb_index and kb_index.is_valid_index(idx):
                topic = kb_index.items[idx]["original_keywords"][0] if kb_index.items[idx]["original_keywords"] else f"Ответ #{idx}"
                keyboard.append([InlineKeyboardButton(f"🔹 {topic}", callback_data=f"fav_show_{idx}")])
        keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_main")])
        return InlineKeyboardMarkup(keyboard)


# ====================== Тексты ======================

class AppleMessages:
    WELCOME = """👋 Привет!

Я — ваш персональный помощник по обучению программированию.

💡 Просто напишите вопрос, и я помогу найти ответ.

👇 Или выберите тему ниже:"""

    WELCOME_RETURNING = """👋 С возвращением!

Чем могу помочь сегодня?"""

    HELP = """📚 <b>Как пользоваться ботом</b>

Просто пишите вопросы на естественном языке — я пойму.

<b>Примеры вопросов:</b>
• «Сколько стоит обучение?»
• «Расскажи о преподавателе»
• «Как записаться на консультацию?»

<b>Возможности:</b>
• Поиск ответов в базе знаний
• Запись на консультацию
• Дорожные карты обучения
• Избранное (сохраняйте полезные ответы)

<i>Я запоминаю контекст беседы, поэтому можно задавать уточняющие вопросы.</i>"""

    NOT_FOUND = """🤔 <b>Пока не знаю ответа</b>

Но я сохранил ваш вопрос — скоро научусь на него отвечать.

<b>Попробуйте:</b>
• Переформулировать вопрос
• Выбрать тему в меню /start
• Написать /help"""

    CONSULTATION_SUCCESS = """✅ <b>Заявка отправлена</b>

Алексей свяжется с вами в ближайшее время.

📅 А пока можете выбрать удобное время в календаре:"""

    FEEDBACK_THANKS = "💚 Спасибо за оценку!"
    FEEDBACK_DISLIKE = "📝 Спасибо за обратную связь. Мы постараемся улучшить ответы."
    CLARIFY_PROMPT = "🤔 Уточните, пожалуйста:"
    FUZZY_SUGGESTION = "💡 Возможно, вы имели в виду:"
    FAVORITE_ADDED = "⭐ Добавлено в избранное"
    FAVORITE_REMOVED = "⭐ Удалено из избранного"
    FAVORITE_EMPTY = "У вас пока нет избранных ответов."


# ====================== Команды ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    cleanup_inactive_users()
    ctx = get_user_context(user_id)
    is_returning = user_id in user_contexts  # проверяем наличие в глобальном словаре
    update_user_activity(user_id)

    text = AppleMessages.WELCOME_RETURNING if is_returning else AppleMessages.WELCOME
    await update.message.reply_text(
        text,
        reply_markup=AppleKeyboards.main_menu(is_returning),
        parse_mode="HTML"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(AppleMessages.HELP, parse_mode="HTML")

async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE,
                           edit_mode: bool = False) -> None:
    text = "🗺 <b>Дорожные карты обучения</b>\n\nВыберите направление:"
    if edit_mode and update.callback_query:
        await update.callback_query.edit_message_text(
            text, reply_markup=AppleKeyboards.roadmaps_menu(), parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            text, reply_markup=AppleKeyboards.roadmaps_menu(), parse_mode="HTML"
        )

async def faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost")],
        [InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")],
        [InlineKeyboardButton("🧠 Метод обучения", callback_data="menu_method")],
        [InlineKeyboardButton("🗓 Консультация", callback_data="menu_consult")],
        [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
    ]
    await update.message.reply_text(
        "📋 <b>Часто задаваемые вопросы</b>\n\nВыберите тему:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )

async def favorites_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    favs = get_favorites(user_id)
    if not favs:
        await update.message.reply_text(
            AppleMessages.FAVORITE_EMPTY,
            reply_markup=AppleKeyboards.back_button()
        )
        return

    keyboard = []
    for idx in favs[:5]:
        if kb_index and kb_index.is_valid_index(idx):
            topic = kb_index.items[idx]["original_keywords"][0] if kb_index.items[idx]["original_keywords"] else f"Ответ #{idx}"
            keyboard.append([InlineKeyboardButton(f"⭐ {topic}", callback_data=f"fav_show_{idx}")])
    keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_main")])

    await update.message.reply_text(
        "⭐ <b>Ваше избранное</b>\n\nНажмите на тему, чтобы увидеть ответ:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


# ====================== Обработчик callback'ов ======================

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data
    update_user_activity(user_id)

    if data == "menu_main":
        await query.edit_message_text(
            AppleMessages.WELCOME_RETURNING,
            reply_markup=AppleKeyboards.main_menu(is_returning=True),
            parse_mode="HTML"
        )
        return

    if data == "menu_consult":
        text = "🗓 <b>Запись на консультацию</b>\n\nВыберите удобный способ:"
        await query.edit_message_text(text, reply_markup=AppleKeyboards.consult_menu(), parse_mode="HTML")
        return

    if data == "menu_roadmaps":
        await roadmaps_command(update, context, edit_mode=True)
        return

    if data == "menu_favorites":
        favs = get_favorites(user_id)
        if not favs:
            await query.edit_message_text(
                AppleMessages.FAVORITE_EMPTY,
                reply_markup=AppleKeyboards.back_button()
            )
            return
        await query.edit_message_text(
            "⭐ <b>Избранное</b>",
            reply_markup=AppleKeyboards.favorites_menu(favs),
            parse_mode="HTML"
        )
        return

    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {
            "menu_cost": "стоимость",
            "menu_method": "метод выстраданного познания",
            "menu_about": "кто такой алексей"
        }
        if not kb_index:
            await query.edit_message_text("⚠️ База знаний недоступна", reply_markup=AppleKeyboards.back_button())
            return

        answer, score, candidates = search_knowledge_base(q_map[data], kb_index)
        if not answer:
            await query.edit_message_text(AppleMessages.NOT_FOUND, reply_markup=AppleKeyboards.back_button(), parse_mode="HTML")
            return

        ans_idx = candidates[0]['index'] if candidates else 0
        save_question_for_answer(user_id, ans_idx, q_map[data])

        clean_text, url_buttons = extract_links_and_buttons(answer.replace("[add_button]", "").strip())
        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])

        url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))

        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    if data.startswith("clarify_"):
        if data == "clarify_none":
            await query.edit_message_text("Хорошо, попробуйте сформулировать иначе.", reply_markup=AppleKeyboards.back_button())
            return

        idx = int(data.split("_")[1])
        if not kb_index or not kb_index.is_valid_index(idx):
            await query.answer("Ответ не найден", show_alert=True)
            return

        context_data = kb_index.items[idx]["context"]
        clean_text, url_buttons = extract_links_and_buttons(context_data.replace("[add_button]", "").strip())
        if "[add_button]" in context_data:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])

        save_question_for_answer(user_id, idx, "Уточняющий вопрос")
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))

        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    if data == "consultation":
        await consultation_callback(update, context)
        return

    if data.startswith("like_") or data.startswith("dislike_"):
        await feedback_callback(update, context)
        return

    if data.startswith("fav_add_"):
        idx = int(data.split("_")[2])
        add_favorite(user_id, idx)
        await query.answer(AppleMessages.FAVORITE_ADDED, show_alert=False)
        return

    if data.startswith("fav_remove_"):
        idx = int(data.split("_")[2])
        remove_favorite(user_id, idx)
        await query.answer(AppleMessages.FAVORITE_REMOVED, show_alert=False)
        return

    if data.startswith("fav_show_"):
        idx = int(data.split("_")[2])
        if not kb_index or not kb_index.is_valid_index(idx):
            await query.answer("Ответ не найден", show_alert=True)
            return

        answer = kb_index.items[idx]["context"]
        clean_text, url_buttons = extract_links_and_buttons(answer.replace("[add_button]", "").strip())
        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])

        url_buttons.append([InlineKeyboardButton("⭐ Удалить из избранного", callback_data=f"fav_remove_{idx}")])
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))

        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return

    if data.startswith("admin_"):
        await admin_callback(update, context)
        return


# ====================== Консультация ======================

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    consultations = load_json(CONSULTATIONS_FILE)
    recent = [c for c in consultations if c.get("user_id") == user.id and
              datetime.now() - datetime.strptime(c.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)]

    if recent:
        await query.edit_message_text(
            "✅ <b>Вы уже записаны</b>\n\nВаша заявка обрабатывается.",
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
            f"🔔 <b>Новая заявка!</b>\n\n👤 {user.first_name}\n📱 @{user.username or 'нет'}\n🆔 {user.id}",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Admin notify error: {e}")

    await query.edit_message_text(
        AppleMessages.CONSULTATION_SUCCESS,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=CALENDAR_URL)]]),
        parse_mode="HTML"
    )


# ====================== Обратная связь ======================

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user = query.from_user

    fb_type = "like" if data.startswith("like_") else "dislike"
    try:
        idx = int(data.split("_")[1])
    except (IndexError, ValueError):
        await query.answer("Ошибка данных", show_alert=True)
        return

    if not kb_index or not kb_index.is_valid_index(idx):
        await query.answer("Ответ не найден", show_alert=True)
        return

    answer = kb_index.items[idx]["context"]
    question = get_question_for_answer(user.id, idx)
    if question == "???":
        ctx = get_user_context(user.id)
        history = list(ctx.get("history", []))
        question = history[-1] if history else "Неизвестный вопрос"

    feedback = load_json(FEEDBACK_FILE)
    feedback.append({
        "type": fb_type,
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "user_id": user.id,
        "username": user.username,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(FEEDBACK_FILE, feedback)

    if fb_type == "like":
        new_kb = InlineKeyboardMarkup([[InlineKeyboardButton("💚 Спасибо за оценку!", callback_data="ignore")]])
        await query.edit_message_reply_markup(new_kb)
    else:
        new_kb = InlineKeyboardMarkup([[InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]])
        await query.edit_message_reply_markup(new_kb)
        await query.message.reply_text(AppleMessages.FEEDBACK_DISLIKE, parse_mode="HTML")

        try:
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"👎 <b>Дизлайк</b>\n\n❓ {question}\n💬 {answer[:100]}...\n👤 @{user.username or user.id}",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Admin notify error: {e}")


# ====================== Админ-панель ======================

async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user_id = update.effective_user.id

    if user_id != ADMIN_USER_ID:
        await query.answer("Доступ запрещён", show_alert=True)
        return

    if data.startswith("admin_page_"):
        parts = data.split("_")
        await admin_show_list(update, context, parts[2], int(parts[3]))
        return

    if data.startswith("admin_clear_"):
        await admin_clear_confirm(update, context, data.replace("admin_clear_", ""))
        return

    if data.startswith("admin_do_clear_"):
        await admin_do_clear(update, context, data.replace("admin_do_clear_", ""))
        return

    if data == "admin_menu_main":
        keyboard = [
            [
                InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"),
                InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")
            ],
            [
                InlineKeyboardButton("❓ Неизвестные", callback_data="admin_page_unknown_0"),
                InlineKeyboardButton("📋 Заявки", callback_data="admin_page_consult_0")
            ],
            [InlineKeyboardButton("📊 Статистика", callback_data="admin_stats")]
        ]
        await query.edit_message_text(
            "<b>📊 Панель управления</b>",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return

    if data == "admin_stats":
        users = len(user_contexts)
        cons = len(load_json(CONSULTATIONS_FILE))
        unk = len(load_json(UNKNOWN_FILE))
        fb = load_json(FEEDBACK_FILE)
        likes = sum(1 for x in fb if x.get("type") == "like")
        dislikes = sum(1 for x in fb if x.get("type") == "dislike")

        text = f"""<b>📈 Статистика</b>

👤 Активных пользователей: {users}
📋 Заявок: {cons}
❓ Неизвестных вопросов: {unk}
👍 Лайков: {likes}
👎 Дизлайков: {dislikes}"""
        await query.edit_message_text(text, reply_markup=AppleKeyboards.back_button("admin_menu_main"), parse_mode="HTML")
        return


async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str, page: int = 0):
    query = update.callback_query
    if query:
        await query.answer()

    items = []
    title = ""
    empty_msg = ""
    clear_callback = ""

    if data_type == "consult":
        items = load_json(CONSULTATIONS_FILE)
        title = "📋 Заявки на консультацию"
        empty_msg = "Заявок пока нет."
        clear_callback = "admin_clear_consult"
    elif data_type == "like":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "like"]
        title = "💚 Лайки"
        empty_msg = "Лайков пока нет."
        clear_callback = "admin_clear_like"
    elif data_type == "dislike":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "dislike"]
        title = "👎 Дизлайки"
        empty_msg = "Жалоб пока нет."
        clear_callback = "admin_clear_dislike"
    elif data_type == "unknown":
        items = load_json(UNKNOWN_FILE)
        title = "❓ Неизвестные вопросы"
        empty_msg = "Неизвестных вопросов нет."
        clear_callback = "admin_clear_unknown"
    else:
        return

    total = len(items)
    total_pages = math.ceil(total / ITEMS_PER_PAGE) if total else 1
    page = max(0, min(page, total_pages - 1))

    text = f"<b>{title}</b>\nВсего: {total}\n\n"
    if not items:
        text += f"<i>{empty_msg}</i>"
    else:
        start = page * ITEMS_PER_PAGE
        for i, item in enumerate(items[start:start+ITEMS_PER_PAGE], start=start+1):
            if data_type == "consult":
                text += f"{i}. {item.get('first_name', '')} @{item.get('username', '')}\n   ⏰ {item.get('timestamp', '')}\n\n"
            elif data_type == "unknown":
                q = item.get('question', '???')
                text += f"{i}. {q[:100]}{'...' if len(q) > 100 else ''}\n\n"
            else:
                q = item.get('question', '???')
                text += f"{i}. {q[:50]}{'...' if len(q) > 50 else ''}\n\n"

    keyboard = []
    if total_pages > 1:
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("◀️", callback_data=f"admin_page_{data_type}_{page-1}"))
        nav_row.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("▶️", callback_data=f"admin_page_{data_type}_{page+1}"))
        keyboard.append(nav_row)

    if items:
        keyboard.append([InlineKeyboardButton("🗑 Очистить", callback_data=clear_callback)])
    keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_menu_main")])

    markup = InlineKeyboardMarkup(keyboard)
    if query:
        await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
    else:
        await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")


async def admin_clear_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    query = update.callback_query
    await query.answer()
    keyboard = [
        [InlineKeyboardButton("✅ Да, очистить", callback_data=f"admin_do_clear_{data_type}")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"admin_page_{data_type}_0")]
    ]
    await query.edit_message_text(
        "⚠️ <b>Подтвердите очистку</b>",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    query = update.callback_query
    await query.answer()

    if data_type == "consult":
        save_json(CONSULTATIONS_FILE, [])
    elif data_type in ["like", "dislike"]:
        fb = load_json(FEEDBACK_FILE)
        save_json(FEEDBACK_FILE, [x for x in fb if x.get("type") != data_type])
    elif data_type == "unknown":
        save_json(UNKNOWN_FILE, [])

    await query.edit_message_text("✅ <b>Очищено успешно</b>", parse_mode="HTML")


# ====================== Обработчик текстовых сообщений ======================

def get_fuzzy_suggestion(question: str) -> Optional[str]:
    if not FUZZY_ENABLED or not kb_index or not kb_index.all_keywords_list:
        return None
    best_match, score = process.extractOne(question, kb_index.all_keywords_list)
    if score > 70:
        return best_match
    return None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    user_question = update.message.text.strip()

    if await handle_admin_text(update, context):
        return

    cleanup_inactive_users()
    ctx = get_user_context(user_id)
    update_user_activity(user_id)
    ctx["history"].append(user_question)

    search_query = get_contextual_question(user_id, user_question)
    answer, score, candidates = search_knowledge_base(search_query, kb_index)
    final_answer = None

    if score > SCORE_DIRECT_ANSWER and answer:
        final_answer = answer
    elif score > SCORE_CLARIFY and candidates:
        keyboard = [[InlineKeyboardButton(f"💬 {c['topic']}", callback_data=f"clarify_{c['index']}")]
                    for c in candidates]
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        await update.message.reply_text(
            AppleMessages.CLARIFY_PROMPT,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return
    elif FUZZY_ENABLED:
        suggestion = get_fuzzy_suggestion(user_question)
        if suggestion:
            answer2, score2, _ = search_knowledge_base(suggestion, kb_index)
            if score2 > SCORE_CLARIFY:
                final_answer = answer2
            if score2 < SCORE_DIRECT_ANSWER and candidates:
                keyboard = [[InlineKeyboardButton(f"💡 {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]]
                await update.message.reply_text(
                    AppleMessages.FUZZY_SUGGESTION,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode="HTML"
                )
                return

    if not final_answer:
        unk = load_json(UNKNOWN_FILE)
        unk.append({
            "question": user_question,
            "user_id": user_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_json(UNKNOWN_FILE, unk)

        await update.message.reply_text(
            AppleMessages.NOT_FOUND,
            reply_markup=AppleKeyboards.main_menu(is_returning=True),
            parse_mode="HTML"
        )
        return

    clean_answer = final_answer.replace("[add_button]", "").strip()
    display_text, url_buttons = extract_links_and_buttons(clean_answer)

    ans_idx = candidates[0]['index'] if candidates and candidates[0]['context'] == final_answer else 0
    if ans_idx == 0:
        for i, item in enumerate(kb_index.items):
            if item['context'] == final_answer:
                ans_idx = i
                break

    save_question_for_answer(user_id, ans_idx, user_question)

    if "[add_button]" in final_answer:
        url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])

    url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))

    await update.message.reply_text(
        display_text,
        reply_markup=InlineKeyboardMarkup(url_buttons),
        disable_web_page_preview=True,
        parse_mode="HTML"
    )


async def handle_admin_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    text = update.message.text.strip().lower()

    if user_id != ADMIN_USER_ID:
        return False

    if text in ["заявки", "заявка", "запись", "записи"]:
        await admin_show_list(update, context, "consult", 0)
        return True

    if text in ["отзыв", "отзывы", "лайки", "дизлайки", "статистика"]:
        keyboard = [
            [
                InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"),
                InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")
            ],
            [
                InlineKeyboardButton("❓ Неизвестные", callback_data="admin_page_unknown_0"),
                InlineKeyboardButton("📋 Заявки", callback_data="admin_page_consult_0")
            ],
            [InlineKeyboardButton("📊 Статистика", callback_data="admin_stats")]
        ]
        await update.message.reply_text(
            "<b>📊 Панель управления</b>",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return True

    if text == "стата":
        users = len(user_contexts)
        cons = len(load_json(CONSULTATIONS_FILE))
        unk = len(load_json(UNKNOWN_FILE))
        fb = load_json(FEEDBACK_FILE)
        likes = sum(1 for x in fb if x.get("type") == "like")
        dislikes = sum(1 for x in fb if x.get("type") == "dislike")
        await update.message.reply_text(
            f"👤 Пользователей: {users}\n📋 Заявок: {cons}\n❓ Неизвестных: {unk}\n👍 Лайков: {likes}\n👎 Дизлайков: {dislikes}"
        )
        return True

    return False


# ====================== Обработчик ошибок ======================

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
            import traceback
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"❌ <b>ERROR:</b>\n<pre>{tb_string[:4000]}</pre>",
                parse_mode="HTML"
            )
        except Exception:
            pass