import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import Update
from config import BOT_TOKEN, logger, VERSION
from utils import initialize_kb
from handlers import start, help_command, handle_message, menu_callback

async def post_init(application: Application):
    await application.bot.delete_webhook(drop_pending_updates=True)
    logger.info("✅ Webhook cleared.")

def main():
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN not found!")
        return
    
    logger.info(f"🚀 Starting bot v{VERSION}...")
    
    # Инициализация базы (Чистое чтение JSON)
    initialize_kb()
    
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async def error_handler(update, context):
        logger.error(f"Error: {context.error}")
    
    application.add_error_handler(error_handler)
    
    logger.info("🤖 Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()