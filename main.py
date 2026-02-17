import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import BOT_TOKEN, logger, VERSION
from utils import initialize_kb
from handlers import (
    start, help_command, handle_message,
    menu_callback, rebuild_keywords_command
)

def main():
    if not BOT_TOKEN:
        logger.error("❌ Токен BOT_TOKEN не найден в .env файле!")
        return
    
    logger.info(f"🚀 Запуск бота v{VERSION}...")
    
    try:
        # ✅ Инициализация базы (ВСЕГДА перегенерирует keywords)
        initialize_kb()
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации базы знаний: {e}")
        return
    
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрация хендлеров
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("rebuild_keywords", rebuild_keywords_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async def error_handler(update, context):
        logger.error(f"Update {update} caused error {context.error}")
    
    application.add_error_handler(error_handler)
    
    logger.info("🤖 Бот начал опрос сервера Telegram")
    application.run_polling()

if __name__ == "__main__":
    main()