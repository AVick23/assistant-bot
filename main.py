import os
import traceback

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler

from config import BOT_TOKEN, ADMIN_USER_ID, logger
from utils import preprocess_knowledge_base, load_knowledge_base, kb_index as kb_global
from handlers import (
    start, help_command, roadmaps_command, faq_command, favorites_command,
    menu_callback, handle_message, error_handler
)


def main() -> None:
    # Загружаем базу знаний
    try:
        kb_raw = load_knowledge_base('main.json')
        global kb_global
        kb_global = preprocess_knowledge_base(kb_raw)
        print(f"✅ База знаний загружена: {len(kb_global.items)} записей")
    except Exception as e:
        print(f"❌ Ошибка загрузки базы знаний: {e}")
        return

    # Создаём приложение
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем хендлеры
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command))
    application.add_handler(CommandHandler("faq", faq_command))
    application.add_handler(CommandHandler("favorites", favorites_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Обработчик ошибок
    application.add_error_handler(error_handler)

    print("🚀 Бот запущен")
    application.run_polling()


if __name__ == "__main__":
    main()