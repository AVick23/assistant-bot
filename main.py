import os
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import Update
from config import BOT_TOKEN, logger, VERSION
from utils import initialize_kb
from handlers import (
    start, help_command, handle_message,
    menu_callback, rebuild_keywords_command
)

# ============================================================
# ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ (СБРОС ВЕБХУКОВ)
# ============================================================
async def post_init(application: Application):
    """Выполняется после инициализации Application, но до старта polling."""
    # Принудительно удаляем вебхук и сбрасываем ожидающие обновления
    # Это решает проблему 409 Conflict, если бот был перезапущен
    await application.bot.delete_webhook(drop_pending_updates=True)
    logger.info("✅ Вебхук сброшен, старые обновления очищены.")

def main():
    if not BOT_TOKEN:
        logger.error("❌ Токен BOT_TOKEN не найден в .env файле!")
        return
    
    logger.info(f"🚀 Запуск бота v{VERSION}...")
    
    try:
        # ✅ Инициализация базы (загружает индекс в память)
        initialize_kb()
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации базы знаний: {e}")
        return
    
    # ✅ Подключаем post_init для очистки
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
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
    
    # drop_pending_updates=True игнорирует сообщения, накопившиеся пока бот был выключен
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()