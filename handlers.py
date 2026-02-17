import math
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from typing import List
from config import (
    ADMIN_USER_ID, FILES, URLS, logger, SETTINGS,
    AppleStyleMessages
)
from utils import (
    kb_index, get_user_context, update_user_activity,
    save_question_for_answer, get_question_for_answer,
    cleanup_inactive_users, extract_links_and_buttons,
    load_json, save_json, user_contexts, initialize_kb,
    save_message_to_history, get_contextual_question
)

# ============================================================
# КЛАВИАТУРЫ (APPLE STYLE)
# ============================================================
class AppleKeyboards:
    @staticmethod
    def main_menu(is_returning: bool = False) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="menu_consult")],
            [
                InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"),
                InlineKeyboardButton("🗺 Карты", callback_data="menu_roadmaps")
            ],
            [
                InlineKeyboardButton("🧠 О методе", callback_data="menu_method"),
                InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")
            ],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def feedback_buttons(ans_idx: int) -> List[List[InlineKeyboardButton]]:
        return [
            [
                InlineKeyboardButton("👍 Полезно", callback_data=f"like_{ans_idx}"),
                InlineKeyboardButton("👎 Не помогло", callback_data=f"dislike_{ans_idx}")
            ]
        ]
    
    @staticmethod
    def consult_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("📅 Выбрать время в календаре", url=URLS['calendar'])],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def roadmaps_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🐍 Python", url=URLS['roadmaps']['python'])],
            [InlineKeyboardButton("⚡ Backend", url=URLS['roadmaps']['backend'])],
            [InlineKeyboardButton("🐹 Golang", url=URLS['roadmaps']['golang'])],
            [InlineKeyboardButton("🔧 DevOps", url=URLS['roadmaps']['devops'])],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def back_button(cb_data="menu_main") -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=cb_data)]])
    
    @staticmethod
    def not_found_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🗓 Связаться с преподавателем", callback_data="menu_consult")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

# ============================================================
# КОМАНДЫ
# ============================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cleanup_inactive_users()
    is_returning = user_id in user_contexts
    
    get_user_context(user_id)
    update_user_activity(user_id)
    
    text = AppleStyleMessages.WELCOME_RETURNING if is_returning else AppleStyleMessages.WELCOME
    await update.message.reply_text(text, reply_markup=AppleKeyboards.main_menu(), parse_mode="HTML")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(AppleStyleMessages.HELP, parse_mode="HTML")

async def rebuild_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Админ-команда для пересборки keywords"""
    user_id = update.effective_user.id
    if user_id != ADMIN_USER_ID:
        return
    
    await update.message.reply_text("🔄 Начинаю перестройку базы знаний...")
    
    try:
        import utils
        updated_count = utils.update_keywords_in_db(force_regenerate=True)
        utils.kb_index = utils.initialize_kb()
        
        await update.message.reply_text(
            f"✅ База знаний обновлена!\n\n"
            f"Записей обновлено: {updated_count}\n"
            f"Индекс перестроен в памяти."
        )
        logger.info(f"Admin {user_id} rebuilt keywords. Updated: {updated_count}")
    except Exception as e:
        logger.error(f"Rebuild error: {e}")
        await update.message.reply_text(f"❌ Ошибка при перестройке: {e}")

# ============================================================
# CALLBACK HANDLER
# ============================================================
async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data
    update_user_activity(user_id)
    
    if data == "menu_main":
        await query.edit_message_text(AppleStyleMessages.WELCOME_RETURNING, reply_markup=AppleKeyboards.main_menu(), parse_mode="HTML")
        return
    
    if data == "menu_consult":
        await query.edit_message_text("🗓 <b>Запись на консультацию</b>", reply_markup=AppleKeyboards.consult_menu(), parse_mode="HTML")
        return
    
    if data == "menu_roadmaps":
        await query.edit_message_text("🗺 <b>Дорожные карты</b>", reply_markup=AppleKeyboards.roadmaps_menu(), parse_mode="HTML")
        return
    
    # Стандартные меню
    menu_map = {
        "menu_cost": "стоимость",
        "menu_method": "метод выстраданного познания",
        "menu_about": "кто такой алексей"
    }
    
    if data in menu_map:
        user_ctx = get_user_context(user_id)
        results = kb_index.search(menu_map[data], user_context=user_ctx)
        
        if results:
            top = results[0]
            ctx_text = top['context']
            clean_text, url_btns = extract_links_and_buttons(ctx_text)
            
            keyboard = []
            for row in url_btns:
                keyboard.append([InlineKeyboardButton(btn['text'], url=btn['url']) for btn in row])
            
            if "[add_button]" in ctx_text:
                keyboard.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
            
            save_question_for_answer(user_id, top['index'], menu_map[data])
            save_message_to_history(user_id, menu_map[data], is_user=True)
            save_message_to_history(user_id, ctx_text[:200], is_user=False)
            keyboard.extend(AppleKeyboards.feedback_buttons(top['index']))
            
            await query.edit_message_text(
                clean_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        else:
            await query.edit_message_text(
                AppleStyleMessages.NOT_FOUND,
                reply_markup=AppleKeyboards.not_found_menu(),
                parse_mode="HTML"
            )
        return
    
    # Уточнение вопроса
    if data.startswith("clarify_"):
        if data == "clarify_none":
            await query.edit_message_text("Хорошо, попробуйте сформулировать иначе.", reply_markup=AppleKeyboards.back_button())
            return
        
        idx = int(data.split("_")[1])
        if not kb_index.is_valid_index(idx):
            await query.answer("Ошибка: ответ не найден.", show_alert=True)
            return
        
        item = kb_index.items[idx]
        ctx_text = item['context']
        clean_text, url_btns = extract_links_and_buttons(ctx_text)
        
        keyboard = []
        for row in url_btns:
            keyboard.append([InlineKeyboardButton(btn['text'], url=btn['url']) for btn in row])
        
        if "[add_button]" in ctx_text:
            keyboard.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        
        save_question_for_answer(user_id, idx, "Уточняющий вопрос")
        save_message_to_history(user_id, "Уточняющий вопрос", is_user=True)
        save_message_to_history(user_id, ctx_text[:200], is_user=False)
        keyboard.extend(AppleKeyboards.feedback_buttons(idx))
        
        await query.edit_message_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        return
    
    # Консультация
    if data == "consultation":
        user = query.from_user
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        consultations = load_json(FILES['consultations'])
        recent = [
            c for c in consultations
            if c.get('user_id') == user.id and
            datetime.now() - datetime.strptime(c.get('timestamp', '2000-01-01'), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)
        ]
        
        if recent:
            await query.edit_message_text(
                "✅ <b>Вы уже записаны</b>\n\nВаша заявка обрабатывается.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=URLS['calendar'])]]),
                parse_mode="HTML"
            )
            return
        
        consultations.append({
            "user_id": user.id,
            "username": user.username or "Нет",
            "first_name": user.first_name or " ",
            "timestamp": timestamp
        })
        save_json(FILES['consultations'], consultations)
        
        try:
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"🔔 <b>Новая заявка!</b>\n👤 {user.first_name}\n📱 @{user.username or 'нет'}",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Admin notify error: {e}")
        
        await query.edit_message_text(
            AppleStyleMessages.CONSULTATION_SUCCESS,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=URLS['calendar'])]]),
            parse_mode="HTML"
        )
        return
    
    # Обратная связь
    if data.startswith("like_") or data.startswith("dislike_"):
        fb_type = "like" if "like_" in data else "dislike"
        try:
            idx = int(data.split("_")[1])
        except ValueError:
            return
        
        if not kb_index.is_valid_index(idx):
            return
        
        question = get_question_for_answer(user.id, idx)
        if question == "???":
            ctx = get_user_context(user.id)
            if ctx['history']:
                question = list(ctx['history'])[-1]
        
        feedback = load_json(FILES['feedback'])
        feedback.append({
            "type": fb_type,
            "question": question,
            "user_id": user.id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_json(FILES['feedback'], feedback)
        
        if fb_type == "like":
            await query.edit_message_reply_markup(
                InlineKeyboardMarkup([[InlineKeyboardButton("💚 Спасибо за оценку!", callback_data="ignore")]])
            )
        else:
            await query.edit_message_reply_markup(
                InlineKeyboardMarkup([[InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]])
            )
            try:
                answer_text = kb_index.items[idx]['context'][:100]
                await context.bot.send_message(
                    ADMIN_USER_ID,
                    f"👎 <b>Дизлайк</b>\n❓ Вопрос: {question}\n💬 Ответ: {answer_text}...",
                    parse_mode="HTML"
                )
            except Exception:
                pass
        return
    
    # Админ-панель
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
    
    if data == "ignore":
        return

# ============================================================
# ГЛАВНЫЙ ОБРАБОТЧИК СООБЩЕНИЙ
# ============================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    
    if await handle_admin_text(update, context):
        return
    
    cleanup_inactive_users()
    ctx = get_user_context(user_id)
    update_user_activity(user_id)
    
    # ✅ Сохраняем вопрос в историю
    save_message_to_history(user_id, user_question, is_user=True)
    
    # ✅ Поиск с учётом контекста беседы
    search_query = get_contextual_question(user_id, user_question)
    results = kb_index.search(search_query, user_context=ctx)
    
    if not results:
        unk = load_json(FILES['unknown'])
        unk.append({
            "question": user_question,
            "user_id": user_id,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_json(FILES['unknown'], unk)
        
        await update.message.reply_text(
            AppleStyleMessages.NOT_FOUND,
            reply_markup=AppleKeyboards.not_found_menu(),
            parse_mode="HTML"
        )
        return
    
    top = results[0]
    min_score = SETTINGS.get('min_bm25_score', 2.5)
    
    if top['score'] > min_score or len(results) == 1:
        final_answer = top['context']
        ans_idx = top['index']
        
        clean_text, url_btns = extract_links_and_buttons(final_answer)
        keyboard = []
        for row in url_btns:
            keyboard.append([InlineKeyboardButton(btn['text'], url=btn['url']) for btn in row])
        
        if "[add_button]" in final_answer:
            keyboard.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        
        save_question_for_answer(user_id, ans_idx, user_question)
        save_message_to_history(user_id, final_answer[:200], is_user=False)
        keyboard.extend(AppleKeyboards.feedback_buttons(ans_idx))
        
        await update.message.reply_text(
            clean_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    else:
        keyboard = []
        for res in results:
            keyboard.append([InlineKeyboardButton(f"💬 {res['topic']}", callback_data=f"clarify_{res['index']}")])
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        
        await update.message.reply_text(
            AppleStyleMessages.CLARIFY_PROMPT,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )

# ============================================================
# АДМИН ФУНКЦИИ
# ============================================================
async def handle_admin_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    text = update.message.text.strip().lower()
    
    if user_id != ADMIN_USER_ID:
        return False
    
    if text in ["заявки", "заявка"]:
        await admin_show_list(update, context, "consult", 0)
        return True
    
    if text in ["статистика", "отзывы"]:
        keyboard = [
            [InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"),
             InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")],
            [InlineKeyboardButton("❓ Неизвестные", callback_data="admin_page_unknown_0")]
        ]
        await update.message.reply_text("<b>📊 Панель управления</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        return True
    
    return False

async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str, page: int = 0):
    query = update.callback_query
    if query:
        await query.answer()
    
    items = []
    title = " "
    
    if data_type == "consult":
        items = load_json(FILES['consultations'])
        title = "📋 Заявки"
    elif data_type == "unknown":
        items = load_json(FILES['unknown'])
        title = "❓ Неизвестные вопросы"
    elif data_type == "like":
        fb = load_json(FILES['feedback'])
        items = [x for x in fb if x['type'] == 'like']
        title = "💚 Лайки"
    elif data_type == "dislike":
        fb = load_json(FILES['feedback'])
        items = [x for x in fb if x['type'] == 'dislike']
        title = "👎 Дизлайки"
    
    total_pages = math.ceil(len(items) / SETTINGS['items_per_page']) if items else 1
    page = max(0, min(page, total_pages - 1))
    
    text = f"<b>{title}</b> (Всего: {len(items)})\n\n"
    
    if items:
        start = page * SETTINGS['items_per_page']
        for i, item in enumerate(items[start:start+SETTINGS['items_per_page']], start+1):
            if data_type == "consult":
                text += f"{i}. {item.get('first_name')} @{item.get('username')}\n"
            else:
                q = item.get('question', '???')
                text += f"{i}. {q[:50]}...\n"
    else:
        text += "<i>Пусто</i>"
    
    keyboard = []
    if total_pages > 1:
        row = []
        if page > 0:
            row.append(InlineKeyboardButton("◀️", callback_data=f"admin_page_{data_type}_{page-1}"))
        row.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            row.append(InlineKeyboardButton("▶️", callback_data=f"admin_page_{data_type}_{page+1}"))
        keyboard.append(row)
    
    if items:
        keyboard.append([InlineKeyboardButton("🗑 Очистить", callback_data=f"admin_clear_{data_type}")])
    
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
    await query.edit_message_text("⚠️ Точно очистить?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    query = update.callback_query
    await query.answer()
    
    if data_type == "consult":
        save_json(FILES['consultations'], [])
    elif data_type == "unknown":
        save_json(FILES['unknown'], [])
    elif data_type in ["like", "dislike"]:
        fb = load_json(FILES['feedback'])
        save_json(FILES['feedback'], [x for x in fb if x['type'] != data_type])
    
    await query.edit_message_text("✅ Очищено.", parse_mode="HTML")