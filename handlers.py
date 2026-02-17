import logging
import traceback
import math
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Импортируем конфиг и утилиты
import config
import utils

logger = logging.getLogger(__name__)


# ============================================================
# 📱 КОМАНДЫ
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start"""
    user_id = update.effective_user.id
    
    # Проверка: новый или возвращающийся пользователь
    is_returning = user_id in utils.user_contexts
    
    # Инициализация контекста
    utils.get_user_context(user_id)
    utils.update_user_activity(user_id)
    
    # ✅ ИСПРАВЛЕНО: Используем config.Messages
    text = config.Messages.WELCOME_RETURNING if is_returning else config.Messages.WELCOME
    
    await update.message.reply_text(
        text, 
        reply_markup=utils.AppleKeyboards.main_menu(user_id),
        parse_mode="HTML"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /help"""
    await update.message.reply_text(
        config.Messages.HELP, 
        parse_mode="HTML"
    )


# ============================================================
# 🎯 ГЛАВНЫЙ ОБРАБОТЧИК CALLBACK-КНОПОК
# ============================================================

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Центральный обработчик всех callback-кнопок"""
    query = update.callback_query
    data = query.data
    
    await query.answer()
    
    user_id = update.effective_user.id
    is_admin = (user_id == config.ADMIN_USER_ID)
    
    utils.update_user_activity(user_id)
    
    # --- АДМИН-ПАНЕЛЬ ---
    
    if data == "admin_menu_main":
        if not is_admin: return
        await query.edit_message_text(
            "🔐 <b>Админ-панель</b>\n\nВыберите действие:",
            reply_markup=utils.AppleKeyboards.admin_panel_main(),
            parse_mode="HTML"
        )
        return

    if data == "admin_stats":
        if not is_admin: return
        fb = utils.load_json(config.FEEDBACK_FILE)
        likes = len([x for x in fb if x.get('type') == 'like'])
        dislikes = len([x for x in fb if x.get('type') == 'dislike'])
        unknowns = len(utils.load_json(config.UNKNOWN_FILE))
        text = (f"📊 <b>Статистика</b>\n\n"
                f"👍 Лайков: {likes}\n"
                f"👎 Дизлайков: {dislikes}\n"
                f"❓ Неизвестных: {unknowns}")
        await query.edit_message_text(text, reply_markup=utils.AppleKeyboards.admin_panel_main(), parse_mode="HTML")
        return

    if data.startswith("admin_page_"):
        if not is_admin: return
        parts = data.split("_")
        # Format: admin_page_type_page (e.g. admin_page_consult_0)
        if len(parts) >= 4:
            data_type = parts[2]
            page = int(parts[3])
            await admin_show_list(update, context, data_type, page)
        return
    
    if data.startswith("admin_clear_"):
        if not is_admin: return
        await admin_clear_confirm(update, context, data.replace("admin_clear_", ""))
        return
    
    if data.startswith("admin_do_clear_"):
        if not is_admin: return
        await admin_do_clear(update, context, data.replace("admin_do_clear_", ""))
        return

    # --- НАВИГАЦИЯ ПОЛЬЗОВАТЕЛЯ ---
    
    if data == "menu_main":
        await query.edit_message_text(
            config.Messages.WELCOME_RETURNING,
            reply_markup=utils.AppleKeyboards.main_menu(user_id),
            parse_mode="HTML"
        )
        return
    
    if data == "menu_consult":
        text = "🗓 <b>Запись на консультацию</b>\n\nВыберите удобный способ:"
        await query.edit_message_text(
            text,
            reply_markup=utils.AppleKeyboards.consult_menu(user_id),
            parse_mode="HTML"
        )
        return
    
    if data == "menu_roadmaps":
        await roadmaps_command(update, context, edit_mode=True)
        return
    
    # --- СТАНДАРТНЫЕ ВОПРОСЫ МЕНЮ ---
    
    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {
            "menu_cost": "стоимость", 
            "menu_method": "метод выстраданного познания", 
            "menu_about": "кто такой алексей"
        }
        
        if not utils.kb_index:
            await query.edit_message_text(
                "⚠️ База знаний недоступна",
                reply_markup=utils.AppleKeyboards.back_button()
            )
            return
        
        answer, score, candidates = utils.search_knowledge_base(q_map[data], utils.kb_index)
        
        if not answer:
            await query.edit_message_text(
                config.Messages.NOT_FOUND,
                reply_markup=utils.AppleKeyboards.back_button(),
                parse_mode="HTML"
            )
            return
        
        # Формируем ответ
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = utils.extract_links_and_buttons(clean_text)
        
        # Определяем индекс ответа
        ans_idx = 0
        if candidates:
            ans_idx = candidates[0]['index']
        else:
            for i, item in enumerate(utils.kb_index.items):
                if item['context'] == answer:
                    ans_idx = i
                    break
        
        # Сохраняем вопрос для этого ответа
        utils.save_question_for_answer(user_id, ans_idx, q_map[data])
        
        # Добавляем кнопку записи, если есть маркер
        if "[add_button]" in answer:
            url_buttons.append([
                InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")
            ])
        
        # Добавляем кнопки обратной связи (только для НЕ админа)
        url_buttons.extend(utils.AppleKeyboards.feedback_buttons(user_id, ans_idx))
        
        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return
    
    # --- УТОЧНЕНИЕ ВОПРОСА ---
    
    if data.startswith("clarify_"):
        if data == "clarify_none":
            await query.edit_message_text(
                "Хорошо, попробуйте сформулировать иначе.",
                reply_markup=utils.AppleKeyboards.back_button()
            )
            return
        
        idx = int(data.split("_")[1])
        
        if not utils.kb_index or not utils.kb_index.is_valid_index(idx):
            await query.answer("Ответ не найден", show_alert=True)
            return
        
        context_data = utils.kb_index.items[idx]["context"]
        clean_text = context_data.replace("[add_button]", "").strip()
        display_text, url_buttons = utils.extract_links_and_buttons(clean_text)
        
        if "[add_button]" in context_data:
            url_buttons.append([
                InlineKeyboardButton("📝 Записаться", callback_data="consultation")
            ])
        
        utils.save_question_for_answer(user_id, idx, "Уточняющий вопрос")
        
        url_buttons.extend(utils.AppleKeyboards.feedback_buttons(user_id, idx))
        
        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        return
    
    # --- КОНСУЛЬТАЦИЯ ---
    
    if data == "consultation":
        await consultation_callback(update, context)
        return
    
    # --- ОБРАТНАЯ СВЯЗЬ ---
    
    if data.startswith("like_") or data.startswith("dislike_"):
        await feedback_callback(update, context)
        return
    
    if data == "ignore":
        return


# ============================================================
# 📝 КОНСУЛЬТАЦИЯ
# ============================================================

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка заявки на консультацию"""
    query = update.callback_query
    user = query.from_user
    
    # Админ не может оставить заявку
    if user.id == config.ADMIN_USER_ID:
        await query.answer("Вы администратор!", show_alert=True)
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Проверка на дубликаты
    consultations = utils.load_json(config.CONSULTATIONS_FILE)
    recent_consultations = [
        c for c in consultations
        if c.get("user_id") == user.id and
        datetime.now() - datetime.strptime(c.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)
    ]
    
    if recent_consultations:
        await query.edit_message_text(
            "✅ <b>Вы уже записаны</b>\n\nВаша заявка обрабатывается. Ожидайте связи!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 Календарь", url=config.CALENDAR_URL)]
            ]),
            parse_mode="HTML"
        )
        return
    
    # Сохранение заявки
    consultations.append({
        "user_id": user.id,
        "username": user.username or "Нет",
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "timestamp": timestamp
    })
    utils.save_json(config.CONSULTATIONS_FILE, consultations)
    
    # ✅ Уведомление админа
    text = (
        f"{config.Messages.ADMIN_NOTIFY_NEW_CONSULT}"
        f"👤 {user.first_name or 'Без имени'}\n"
        f"📱 @{user.username or 'нет username'}\n"
        f"🆔 {user.id}"
    )
    await utils.notify_admin(context, text)
    
    # Ответ пользователю
    keyboard = [[InlineKeyboardButton("📅 Выбрать время в календаре", url=config.CALENDAR_URL)]]
    await query.edit_message_text(
        config.Messages.CONSULTATION_SUCCESS,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


# ============================================================
# 💚 ОБРАТНАЯ СВЯЗЬ
# ============================================================

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка лайков/дизлайков"""
    query = update.callback_query
    data = query.data
    user = query.from_user
    
    # Админ не может оценивать
    if user.id == config.ADMIN_USER_ID:
        await query.answer("Админ не может оценивать.", show_alert=True)
        return

    fb_type = "like" if data.startswith("like_") else "dislike"
    
    try:
        idx = int(data.split("_")[1])
    except (IndexError, ValueError) as e:
        logger.error(f"Invalid callback data format: {data}")
        await query.answer("Ошибка данных", show_alert=True)
        return
    
    if not utils.kb_index or not utils.kb_index.is_valid_index(idx):
        logger.error(f"Index {idx} out of bounds")
        await query.answer("Ответ не найден", show_alert=True)
        return
    
    answer = utils.kb_index.items[idx]["context"]
    question = utils.get_question_for_answer(user.id, idx)
    
    # Fallback на последний вопрос
    if question == "???":
        ctx = utils.get_user_context(user.id)
        history = ctx.get("history", [])
        if history:
            question = list(history)[-1]
    
    # Сохранение фидбека
    feedback_list = utils.load_json(config.FEEDBACK_FILE)
    feedback_list.append({
        "type": fb_type,
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "user_id": user.id,
        "username": user.username,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    utils.save_json(config.FEEDBACK_FILE, feedback_list)
    
    # Визуальная обратная связь и уведомление админа
    if fb_type == "like":
        # Уведомление админа
        msg_text = f"{config.Messages.ADMIN_NOTIFY_NEW_LIKE}❓ {question}\n👤 @{user.username or user.id}"
        await utils.notify_admin(context, msg_text)
        
        new_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💚 Спасибо за оценку!", callback_data="ignore")]
        ])
        await query.edit_message_reply_markup(new_keyboard)
    else:
        # Уведомление админа
        msg_text = f"{config.Messages.ADMIN_NOTIFY_NEW_DISLIKE}❓ {question}\n💬 {answer[:100]}...\n👤 @{user.username or user.id}"
        await utils.notify_admin(context, msg_text)
        
        new_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]
        ])
        await query.edit_message_reply_markup(new_keyboard)
        await query.message.reply_text(config.Messages.FEEDBACK_DISLIKE, parse_mode="HTML")


# ============================================================
# 💬 ГЛАВНЫЙ ОБРАБОТЧИК СООБЩЕНИЙ
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка текстовых сообщений"""
    
    if not update.message or not update.message.text:
        return
    
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    
    # Проверка админ-команд
    if await handle_admin_text(update, context):
        return
    
    utils.update_user_activity(user_id)
    ctx = utils.get_user_context(user_id)
    ctx["history"].append(user_question)
    
    # Поиск с учётом контекста
    search_query = utils.get_contextual_question(user_id, user_question)
    answer, score, candidates = utils.search_knowledge_base(search_query, utils.kb_index)
    final_answer = None
    
    # Логика выбора ответа
    if score > 3.5 and answer:
        final_answer = answer
    elif score > 1.5 and candidates:
        # Предложение уточнения
        keyboard = [
            [InlineKeyboardButton(f"💬 {c['topic']}", callback_data=f"clarify_{c['index']}")]
            for c in candidates
        ]
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        
        await update.message.reply_text(
            config.Messages.CLARIFY_PROMPT,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return
    elif utils.FUZZY_ENABLED:
        # Попытка исправить опечатку
        suggestion = utils.get_fuzzy_suggestion(user_question, utils.kb_index)
        if suggestion:
            answer, score, candidates = utils.search_knowledge_base(suggestion, utils.kb_index)
            if score > 1.5:
                final_answer = answer
            if score < 3.5 and candidates:
                keyboard = [
                    [InlineKeyboardButton(f"💡 {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]
                ]
                await update.message.reply_text(
                    config.Messages.FUZZY_SUGGESTION,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode="HTML"
                )
                return
    
    # Ответ не найден
    if not final_answer:
        # Сохранение неизвестного вопроса
        unk = utils.load_json(config.UNKNOWN_FILE)
        unk.append({
            "question": user_question,
            "user_id": user_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        utils.save_json(config.UNKNOWN_FILE, unk)
        
        # ✅ Уведомление админа
        text = f"{config.Messages.ADMIN_NOTIFY_UNKNOWN}❓ {user_question}\n🆔 {user_id}"
        await utils.notify_admin(context, text)
        
        await update.message.reply_text(
            config.Messages.NOT_FOUND,
            reply_markup=utils.AppleKeyboards.main_menu(user_id),
            parse_mode="HTML"
        )
        return
    
    # Формирование ответа
    clean_answer = final_answer.replace("[add_button]", "").strip()
    ctx["last_answer"] = clean_answer
    
    display_text, url_buttons = utils.extract_links_and_buttons(clean_answer)
    
    # Определение индекса ответа
    ans_idx = 0
    if candidates and candidates[0]['context'] == final_answer:
        ans_idx = candidates[0]['index']
    else:
        for i, item in enumerate(utils.kb_index.items):
            if item['context'] == final_answer:
                ans_idx = i
                break
    
    # Сохраняем вопрос для этого ответа
    utils.save_question_for_answer(user_id, ans_idx, user_question)
    
    # Добавление кнопок
    if "[add_button]" in final_answer:
        url_buttons.append([
            InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")
        ])
    
    # Добавляем кнопки оценки (для админа пусто)
    url_buttons.extend(utils.AppleKeyboards.feedback_buttons(user_id, ans_idx))
    
    await update.message.reply_text(
        display_text,
        reply_markup=InlineKeyboardMarkup(url_buttons),
        disable_web_page_preview=True,
        parse_mode="HTML"
    )


# ============================================================
# 👨‍💼 АДМИН-ПАНЕЛЬ (ТЕКСТ + СПИСКИ)
# ============================================================

async def handle_admin_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обработка текстовых команд админа"""
    user_id = update.effective_user.id
    text = update.message.text.strip().lower()
    
    if user_id != config.ADMIN_USER_ID:
        return False
    
    if text in ["заявки", "заявка", "запись", "записи"]:
        await admin_show_list(update, context, "consult", 0)
        return True
    
    if text in ["отзыв", "отзывы", "статистика"]:
        await update.message.reply_text(
            "📊 Выберите раздел:",
            reply_markup=utils.AppleKeyboards.admin_panel_main(),
            parse_mode="HTML"
        )
        return True
    
    return False


async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                          data_type: str, page: int = 0):
    """Отображение списка данных для админа"""
    query = update.callback_query
    if query:
        await query.answer()
    
    items = []
    title = ""
    empty_msg = ""
    clear_callback = ""
    
    if data_type == "consult":
        items = utils.load_json(config.CONSULTATIONS_FILE)
        title = "📋 Заявки на консультацию"
        empty_msg = "Заявок пока нет."
        clear_callback = "admin_clear_consult"
    elif data_type == "like":
        all_fb = utils.load_json(config.FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "like"]
        title = "💚 Лайки"
        empty_msg = "Лайков пока нет."
        clear_callback = "admin_clear_like"
    elif data_type == "dislike":
        all_fb = utils.load_json(config.FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "dislike"]
        title = "👎 Дизлайки"
        empty_msg = "Жалоб пока нет."
        clear_callback = "admin_clear_dislike"
    elif data_type == "unknown":
        items = utils.load_json(config.UNKNOWN_FILE)
        title = "❓ Неизвестные вопросы"
        empty_msg = "Бот знает ответы на все вопросы."
        clear_callback = "admin_clear_unknown"
    
    total_items = len(items)
    total_pages = math.ceil(total_items / config.ITEMS_PER_PAGE) if total_items > 0 else 1
    
    if page < 0: page = 0
    if page >= total_pages: page = total_pages - 1
    
    text = f"<b>{title}</b>\nВсего: {total_items}\n\n"
    
    if not items:
        text += f"<i>{empty_msg}</i>"
    else:
        start_idx = page * config.ITEMS_PER_PAGE
        end_idx = start_idx + config.ITEMS_PER_PAGE
        current_items = items[start_idx:end_idx]
        
        for i, item in enumerate(current_items, start=start_idx + 1):
            if data_type == "consult":
                text += f"{i}. {item.get('first_name', '')} @{item.get('username', '')}\n   ⏰ {item.get('timestamp', '')}\n\n"
            elif data_type == "unknown":
                text += f"{i}. {item.get('question', '???')}\n\n"
            else:
                q = item.get('question', '???')
                text += f"{i}. {q[:50]}{'...' if len(q) > 50 else ''}\n\n"
    
    keyboard = []
    
    # Навигация
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
    
    keyboard.append([InlineKeyboardButton("🔙 В админ-меню", callback_data="admin_menu_main")])
    
    markup = InlineKeyboardMarkup(keyboard)
    
    if query:
        try:
            await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except Exception:
            pass
    else:
        await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")


async def admin_clear_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    """Подтверждение очистки"""
    query = update.callback_query
    
    keyboard = [
        [InlineKeyboardButton("✅ Да, очистить", callback_data=f"admin_do_clear_{data_type}")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"admin_page_{data_type}_0")]
    ]
    
    await query.edit_message_text(
        "⚠️ <b>Подтвердите очистку</b>\n\nЭто действие нельзя отменить.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    """Выполнение очистки"""
    query = update.callback_query
    
    if data_type == "consult":
        utils.save_json(config.CONSULTATIONS_FILE, [])
    elif data_type in ["like", "dislike"]:
        fb = utils.load_json(config.FEEDBACK_FILE)
        utils.save_json(config.FEEDBACK_FILE, [x for x in fb if x.get("type") != data_type])
    elif data_type == "unknown":
        utils.save_json(config.UNKNOWN_FILE, [])
    
    await query.edit_message_text("✅ <b>Очищено успешно</b>", parse_mode="HTML")


# ============================================================
# ⚠️ ОБРАБОТЧИК ОШИБОК
# ============================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок"""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    # Уведомление админа
    if config.ADMIN_USER_ID:
        try:
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            
            await context.bot.send_message(
                config.ADMIN_USER_ID,
                f"❌ <b>ERROR:</b>\n<pre>{tb_string[:4000]}</pre>",
                parse_mode="HTML"
            )
        except Exception:
            pass