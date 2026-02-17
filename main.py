# main.py
import logging
import os
import sys

from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler

from config import BOT_TOKEN, MAIN_JSON
from utils import load_json, preprocess_knowledge_base
from handlers import (
    start, help_command, menu_callback,
    handle_message, error_handler, handle_add_answer
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main() -> None:
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set")
        sys.exit(1)

    # Загрузка БЗ
    try:
        kb_data = load_json(MAIN_JSON)
        if not kb_data:
            logger.error(f"Knowledge base empty")
            sys.exit(1)
        kb_index = preprocess_knowledge_base(kb_data)
        logger.info(f"Knowledge base loaded: {len(kb_index.items)} entries")
    except Exception as e:
        logger.error(f"Failed to load KB: {e}")
        sys.exit(1)

    application = Application.builder().token(BOT_TOKEN).build()
    application.bot_data['kb_index'] = kb_index

    # Хендлеры
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    
    # Сначала пытаемся добавить ответ (если админ), потом обычные сообщения
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_answer), group=1)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message), group=2)

    application.add_error_handler(error_handler)

    logger.info("Bot started (Apple Style)")
    application.run_polling()

if __name__ == "__main__":
    main()