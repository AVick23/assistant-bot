# messages.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List, Tuple, Optional
from config import ADMIN_USER_ID, CALENDAR_URL


class AppleStyleMessages:
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
• История ваших вопросов

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

    FEEDBACK_THANKS = """💚 Спасибо за оценку!

Ваше мнение помогает становиться лучше."""

    FEEDBACK_DISLIKE = """📝 Спасибо за обратную связь

Ваш отзыв отправлен разработчику. Мы постараемся улучшить ответы."""

    CLARIFY_PROMPT = """🤔 Уточните, пожалуйста:"""

    FUZZY_SUGGESTION = """💡 Возможно, вы имели в виду:"""

    HISTORY_EMPTY = """📭 <b>История пуста</b>

Вы ещё не задавали вопросов."""

    HISTORY_TITLE = """📋 <b>Ваша история</b>

Последние {count} диалогов:"""

    FAQ_TITLE = """❓ <b>Часто задаваемые вопросы</b>

Выберите тему:"""

    FEEDBACK_PROMPT = """💬 <b>Оставить отзыв о боте</b>

Напишите ваше сообщение, и оно будет отправлено разработчику."""

    FEEDBACK_SENT = """✅ Спасибо! Ваш отзыв отправлен."""

    ADMIN_PANEL_TITLE = """🛠 <b>Админ-панель</b>

Выберите раздел:"""

    STATS_TITLE = """📊 <b>Статистика</b>

👥 Пользователей всего: {total_users}
✨ Активных за 24ч: {active_day}
📆 Активных за неделю: {active_week}
📝 Всего вопросов (история): {total_questions}
"""

    ADD_ANSWER_PROMPT = """📝 <b>Добавить ответ на вопрос</b>

Вопрос: <i>{question}</i>

Отправьте текст ответа (можно использовать [add_button] для кнопки записи):"""

    ANSWER_ADDED = """✅ Ответ успешно добавлен в базу знаний!"""


class AppleKeyboards:
    @staticmethod
    def main_menu(is_returning: bool = False, is_admin: bool = False) -> InlineKeyboardMarkup:
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
            [
                InlineKeyboardButton("📋 История", callback_data="menu_history"),
                InlineKeyboardButton("💬 Отзыв", callback_data="menu_feedback")
            ],
            [InlineKeyboardButton("❓ FAQ", callback_data="menu_faq")],
        ]
        if is_admin:
            keyboard.append([InlineKeyboardButton("🛠 Админ-панель", callback_data="admin_menu")])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def feedback_buttons(answer_index: int) -> List[List[InlineKeyboardButton]]:
        return [
            [
                InlineKeyboardButton("👍 Полезно", callback_data=f"like_{answer_index}"),
                InlineKeyboardButton("👎 Не помогло", callback_data=f"dislike_{answer_index}")
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
            [InlineKeyboardButton("🐍 Python", url="https://avick23.github.io/roadmap_python/")],
            [InlineKeyboardButton("⚡ Backend", url="https://avick23.github.io/roadmap_backend/")],
            [InlineKeyboardButton("🐹 Golang", url="https://avick23.github.io/roadmap_golang/")],
            [InlineKeyboardButton("🔧 DevOps", url="https://avick23.github.io/roadmap_devops/")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def back_button(callback_data: str = "menu_main") -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=callback_data)]])

    @staticmethod
    def admin_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📋 Заявки", callback_data="admin_consult_0"),
                InlineKeyboardButton("💚 Лайки", callback_data="admin_like_0")
            ],
            [
                InlineKeyboardButton("👎 Дизлайки", callback_data="admin_dislike_0"),
                InlineKeyboardButton("❓ Неизвестные", callback_data="admin_unknown_0")
            ],
            [InlineKeyboardButton("📊 Статистика", callback_data="admin_stats")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def admin_item_actions(item_type: str, item_index: int, page: int, can_add: bool = False) -> InlineKeyboardMarkup:
        keyboard = []
        if can_add:
            keyboard.append([InlineKeyboardButton("➕ Добавить ответ", callback_data=f"admin_add_{item_type}_{item_index}")])
        keyboard.append([
            InlineKeyboardButton("🗑 Очистить всё", callback_data=f"admin_clear_{item_type}"),
            InlineKeyboardButton("◀️ Назад", callback_data=f"admin_{item_type}_{page}")
        ])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def pagination(base_callback: str, page: int, total_pages: int) -> List[InlineKeyboardButton]:
        row = []
        if page > 0:
            row.append(InlineKeyboardButton("◀️", callback_data=f"{base_callback}_{page-1}"))
        row.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            row.append(InlineKeyboardButton("▶️", callback_data=f"{base_callback}_{page+1}"))
        return row

    @staticmethod
    def confirm_clear(item_type: str, page: int) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("✅ Да, очистить", callback_data=f"admin_do_clear_{item_type}")],
            [InlineKeyboardButton("❌ Отмена", callback_data=f"admin_{item_type}_{page}")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def faq_menu(faq_items: List[Tuple[str, int]]) -> InlineKeyboardMarkup:
        keyboard = []
        for title, idx in faq_items:
            keyboard.append([InlineKeyboardButton(title, callback_data=f"faq_{idx}")])
        keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_main")])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def history_menu(history: List[Tuple[str, str]], page: int, total_pages: int) -> InlineKeyboardMarkup:
        keyboard = []
        # В истории можно показывать только кнопки навигации, т.к. текст длинный
        if total_pages > 1:
            nav_row = AppleKeyboards.pagination("history_page", page, total_pages)
            keyboard.append(nav_row)
        keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_main")])
        return InlineKeyboardMarkup(keyboard)