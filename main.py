from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler

from config import BOT_TOKEN, logger
from utils import init_knowledge_base
from handlers import (
    start, help_command, roadmaps_command, faq_command, favorites_command,
    menu_callback, handle_message, error_handler
)


def main() -> None:
    # Инициализация базы знаний
    try:
        init_knowledge_base('main.json')
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки базы знаний: {e}")
        return

    # Создание приложения
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрация хендлеров
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command))
    application.add_handler(CommandHandler("faq", faq_command))
    application.add_handler(CommandHandler("favorites", favorites_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(error_handler)

    logger.info("🚀 Бот запущен")
    application.run_polling()


if __name__ == "__main__":
    main()