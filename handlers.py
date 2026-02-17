import math
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from typing import List
from config import ADMIN_USER_ID, FILES, URLS, logger, SETTINGS, AppleStyleMessages
from utils import (
    get_kb_index, get_user_context, update_user_activity,
    save_question_for_answer, get_question_for_answer,
    cleanup_inactive_users, extract_links_and_buttons,
    load_json, save_json, user_contexts, 
    save_message_to_history, get_contextual_question
)

class AppleKeyboards:
    @staticmethod
    def main_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🗓 Записаться", callback_data="menu_consult")],
            [
                InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"),
                InlineKeyboardButton("🗺 Карты", callback_data="menu_roadmaps")
            ],
            [InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")],
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def feedback_buttons(ans_idx: int) -> List[List[InlineKeyboardButton]]:
        return [[
            InlineKeyboardButton("👍", callback_data=f"like_{ans_idx}"),
            InlineKeyboardButton("👎", callback_data=f"dislike_{ans_idx}")
        ]]
    
    @staticmethod
    def consult_menu() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("📅 Календарь", url=URLS['calendar'])],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ])
    
    @staticmethod
    def roadmaps_menu() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🐍 Python", url=URLS['roadmaps']['python'])],
            [InlineKeyboardButton("⚡ Backend", url=URLS['roadmaps']['backend'])],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ])

    @staticmethod
    def not_found_menu() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🗓 Связаться", callback_data="menu_consult")],
            [InlineKeyboardButton("🏠 Меню", callback_data="menu_main")]
        ])

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

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data
    update_user_activity(user_id)
    kb_index = get_kb_index()
    
    if data == "menu_main":
        return await query.edit_message_text(AppleStyleMessages.WELCOME_RETURNING, reply_markup=AppleKeyboards.main_menu(), parse_mode="HTML")
    if data == "menu_consult":
        return await query.edit_message_text("🗓 <b>Запись на консультацию</b>", reply_markup=AppleKeyboards.consult_menu(), parse_mode="HTML")
    if data == "menu_roadmaps":
        return await query.edit_message_text("🗺 <b>Карты развития</b>", reply_markup=AppleKeyboards.roadmaps_menu(), parse_mode="HTML")
    
    menu_map = {
        "menu_cost": "стоимость обучения",
        "menu_method": "метод выстраданного познания",
        "menu_about": "кто такой алексей"
    }
    
    if data in menu_map:
        user_ctx = get_user_context(user_id)
        results = kb_index.search(menu_map[data], user_context=user_ctx)
        if results:
            top = results[0]
            clean_text, url_btns = extract_links_and_buttons(top['context'])
            keyboard = [[InlineKeyboardButton(b['text'], url=b['url'])] for b in url_btns]
            if "[add_button]" in top['context']:
                keyboard.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
            keyboard.extend(AppleKeyboards.feedback_buttons(top['index']))
            
            save_question_for_answer(user_id, top['index'], menu_map[data])
            return await query.edit_message_text(clean_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML", disable_web_page_preview=True)
        else:
            return await query.edit_message_text(AppleStyleMessages.NOT_FOUND, reply_markup=AppleKeyboards.not_found_menu(), parse_mode="HTML")

    if data == "consultation":
        user = query.from_user
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        consultations = load_json(FILES['consultations'])
        consultations.append({"user_id": user.id, "username": user.username or "Нет", "first_name": user.first_name or " ", "timestamp": timestamp})
        save_json(FILES['consultations'], consultations)
        try:
            await context.bot.send_message(ADMIN_USER_ID, f"🔔 <b>Заявка!</b>\n👤 {user.first_name}\n📱 @{user.username or 'нет'}", parse_mode="HTML")
        except: pass
        return await query.edit_message_text(AppleStyleMessages.CONSULTATION_SUCCESS, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📅 Календарь", url=URLS['calendar'])]]), parse_mode="HTML")

    if data.startswith("like_") or data.startswith("dislike_"):
        # Логика лайков (как раньше)
        pass 
    if data == "ignore":
        pass

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    
    cleanup_inactive_users()
    ctx = get_user_context(user_id)
    update_user_activity(user_id)
    kb_index = get_kb_index()
    
    save_message_to_history(user_id, user_question, is_user=True)
    
    search_query = get_contextual_question(user_id, user_question)
    results = kb_index.search(search_query, user_context=ctx)
    
    # ✅ ИСПРАВЛЕНИЕ: Если результатов нет ИЛИ верхний результат имеет низкий скор — говорим "Не знаю"
    if not results:
        unk = load_json(FILES['unknown'])
        unk.append({"question": user_question, "user_id": user_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        save_json(FILES['unknown'], unk)
        return await update.message.reply_text(AppleStyleMessages.NOT_FOUND, reply_markup=AppleKeyboards.not_found_menu(), parse_mode="HTML")
    
    # Берем топ результат
    top = results[0]
    
    # Если есть явный лидер (скор > 1.0 - это значит было совпадение по keywords или сильное по BM25)
    # Либо если результат всего один
    if top['score'] > 1.0 or len(results) == 1:
        clean_text, url_btns = extract_links_and_buttons(top['context'])
        keyboard = [[InlineKeyboardButton(b['text'], url=b['url'])] for b in url_btns]
        if "[add_button]" in top['context']:
            keyboard.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        
        save_question_for_answer(user_id, top['index'], user_question)
        save_message_to_history(user_id, top['context'][:100], is_user=False)
        keyboard.extend(AppleKeyboards.feedback_buttons(top['index']))
        
        return await update.message.reply_text(clean_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML", disable_web_page_preview=True)
    else:
        # Если есть несколько слабых кандидатов — предлагаем уточнить
        keyboard = []
        for res in results[:3]:
            keyboard.append([InlineKeyboardButton(f"💬 {res['topic']}", callback_data=f"clarify_{res['index']}")])
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        return await update.message.reply_text(AppleStyleMessages.CLARIFY_PROMPT, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")