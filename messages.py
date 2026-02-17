# messages.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List, Tuple, Optional

from config import ADMIN_USER_ID, CALENDAR_URL

class AppleStyleMessages:
    WELCOME = """👋 Привет!

Я помогаю освоить программирование.

Просто напишите вопрос или выберите тему ниже:"""

    WELCOME_RETURNING = """👋 С возвращением!

Чем могу помочь?"""

    HELP = """📚 <b>Как это работает</b>

Просто пишите вопросы текстом — я пойму контекст.

<b>Примеры:</b>
• «Сколько стоит?»
• «Кто преподает?»
• «Как начать учить Python?»

Работаю 24/7, помню нашу переписку."""

    NOT_FOUND = """🤔 <b>Пока не знаю ответа</b>

Я сохранил ваш вопрос и отправил уведомление.

Если вопрос срочный — запишитесь на консультацию, там помогут точно."""

    CONSULTATION_SUCCESS = """✅ <b>Заявка отправлена</b>

Алексей свяжется с вами в ближайшее время.

📅 А пока можете выбрать удобное время в календаре:"""

    FEEDBACK_THANKS = """💚 Спасибо за оценку!

Ваше мнение помогает становиться лучше."""

    FEEDBACK_DISLIKE = """📝 Спасибо за обратную связь.

Я передал информацию разработчику."""

    # Используется, если бот почти уверен, но хочет подстраховаться
    CLARIFY_PROMPT = """🤔 Уточните, пожалуйста:"""

    HISTORY_EMPTY = """📭 <b>История пуста</b>

Вы ещё не задавали вопросов."""

    HISTORY_TITLE = """📋 <b>Ваша история</b>

Последние {count} диалогов:"""

    ADMIN_PANEL_TITLE = """🛠 <b>Админ-панель</b>

Выберите раздел:"""

    STATS_TITLE = """📊 <b>Статистика</b>

👥 Пользователей всего: {total_users}
✨ Активных за 24ч: {active_day}
📆 Активных за неделю: {active_week}
📝 Всего вопросов: {total_questions}
"""

    ADD_ANSWER_PROMPT = """📝 <b>Добавить ответ</b>

Вопрос: <i>{question}</i>

Отправьте текст ответа:"""

    ANSWER_ADDED = """✅ Ответ успешно добавлен!"""

class AppleKeyboards:
    @staticmethod
    def main_menu(is_returning: bool = False, is_admin: bool = False) -> InlineKeyboardMarkup:
        # Философия Apple: только самые важные действия
        keyboard = [
            [InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="menu_consult")],
            [
                InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"),
                InlineKeyboardButton("🗺 Карты", callback_data="menu_roadmaps")
            ],
        ]
        if is_admin:
            keyboard.append([InlineKeyboardButton("🛠 Админ-панель", callback_data="admin_menu")])
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def feedback_buttons(answer_index: int) -> List[List[InlineKeyboardButton]]:
        return [
            [
                InlineKeyboardButton("👍", callback_data=f"like_{answer_index}"),
                InlineKeyboardButton("👎", callback_data=f"dislike_{answer_index}")
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
                InlineKeyboardButton("❓ Вопросы", callback_data="admin_unknown_0")
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
            InlineKeyboardButton("🗑 Очистить", callback_data=f"admin_clear_{item_type}"),
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

    # Новая клавиатура для уточнения (если бот не уверен)
    @staticmethod
    def clarification_menu(candidates: List[dict]) -> InlineKeyboardMarkup:
        keyboard = []
        # Показываем топ-3 варианта
        for c in candidates[:3]:
            keyboard.append([InlineKeyboardButton(f"💬 {c['topic']}", callback_data=f"clarify_{c['index']}")])
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        return InlineKeyboardMarkup(keyboard)