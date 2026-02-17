import logging
import traceback
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler

import config
import utils
import handlers

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main() -> None:
    # Проверка токена
    if not config.BOT_TOKEN:
        raise ValueError("❌ Токен не найден в переменных окружения BOT_TOKEN")
    
    # Загрузка базы знаний
    try:
        raw_kb = utils.load_knowledge_base(config.KB_FILE)
        processed_kb = utils.preprocess_knowledge_base(raw_kb)
        
        # Важно: присваиваем в модуль utils, чтобы handlers имели к ней доступ
        utils.kb_index = processed_kb
        
        print(f"✅ База знаний загружена: {len(utils.kb_index.items)} записей")
    except Exception as e:
        print(f"❌ Ошибка загрузки базы знаний: {str(e)}")
        return
    
    # Создание приложения
    application = Application.builder().token(config.BOT_TOKEN).build()
    
    # Регистрация хендлеров
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CallbackQueryHandler(handlers.menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    
    # Обработчик ошибок
    application.add_error_handler(handlers.error_handler)
    
    print("🚀 Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()