import json
import re
import numpy as np
import warnings
from typing import Dict, List, Set, Optional, Tuple
import math
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Загрузка переменных окружения
load_dotenv()

import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Константы
ADMIN_USER_ID = 1373472999
CONSULTATIONS_FILE = "consultations.json"
CALENDAR_URL = "https://calendar.app.google/ThpteAc5uqhxqnUA9"
SITE_URL = "https://avick23.github.io/Business-card/"

# Инициализация анализатора
morph = pymorphy2.MorphAnalyzer()

# Расширенный список стоп-слов
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
    'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'уже', 'всего', 'всё',
    # ... (остальной список стоп-слов можно оставить как был) ...
    'быть', 'будет', 'сказал', 'этот', 'это', 'здесь', 'тот', 'там', 'где',
    'который', 'которая', 'которые', 'их', 'этого', 'этой', 'этому', 'этим',
    'эти', 'этих', 'ваш', 'ваша', 'ваше', 'вашего', 'вашей', 'какой', 'какая',
    'какое', 'какие', 'какого', 'каком', 'какими', 'мы', 'наш', 'наша', 'наше',
    'мой', 'моя', 'моё', 'мои', 'твой', 'твоя', 'твоё', 'твои', 'сам', 'сама',
    'само', 'сами', 'тот', 'та', 'то', 'те', 'чей', 'чья', 'чьё', 'чьи', 'кто',
    'что', 'где', 'куда', 'откуда', 'когда', 'почему', 'зачем', 'как', 'либо',
    'нибудь', 'также', 'потому', 'чтобы', 'который', 'свой', 'своя', 'своё',
    'свои', 'самый', 'самая', 'самое', 'самые', 'или', 'ну', 'эх', 'ах', 'ох',
    'без', 'над', 'под', 'перед', 'после', 'между', 'через', 'чтобы', 'ради',
    'для', 'до', 'после', 'около', 'возле', 'рядом', 'мимо', 'вокруг', 'против',
    'за', 'надо', 'нужно', 'может', 'можно', 'должен', 'должна', 'должно',
    'должны', 'хочу', 'хочешь', 'хочет', 'хотим', 'хотите', 'хотят', 'буду',
    'будешь', 'будет', 'будем', 'будете', 'будут', 'хотя', 'если', 'пока',
    'чтоб', 'зато', 'итак', 'также', 'тоже'
}

# Расширенный словарь синонимов с учетом твоей экосистемы
SYNONYMS = {
    'стоимость': ['цена', 'тариф', 'плата', 'расценка', 'сколько стоит'],
    'курс': ['обучение', 'программа', 'тренинг', 'обучение'],
    'преподаватель': ['учитель', 'репетитор', 'тренер', 'лектор', 'алексей', 'avick23'],
    'занятие': ['урок', 'лекция', 'пара', 'встреча'],
    'группа': ['команда', 'коллектив', 'мини-группа'],
    'метод': ['подход', 'техника', 'стратегия', 'выстраданного познания', 'система'],
    'домашка': ['задание', 'дз', 'практика'],
    'бот': ['чат-бот', 'ассистент', 'помощник', 'прогресс', 'прогрессбот', 'прогресс бот'],
    'python': ['питон', 'пайтон'],
    'программирование': ['кодинг', 'разработка', 'it'],
    'вопрос': ['запрос', 'проблема', 'тема'],
    'ответ': ['решение', 'отклик'],
    'начать': ['стартовать', 'приступить'],
    'записаться': ['зарегистрироваться', 'подписаться', 'хочу учиться'],
    'сложный': ['трудный', 'замысловатый', 'запутанный'],
    'легкий': ['простой', 'нетрудный'],
    'быстро': ['скорость', 'оперативно', 'в срок'],
    'долго': ['медленно', 'затянуто'],
    'качество': ['уровень', 'стандарт'],
    'консультация': ['встреча', 'совет', 'помощь', 'бесплатная встреча'],
    'доступ': ['получение', 'возможность'],
    'материалы': ['уроки', 'лекции', 'ресурсы', 'дорожная карта', 'roadmap'],
    'поддержка': ['помощь', 'сопровождение', 'причал', 'сообщество'],
    'экосистема': ['система', 'прогресс', 'прогресс+', 'прогресс плюс'],
    'причал': ['сообщество', 'чат', 'поддержка'],
    'roadmap': ['дорожная карта', 'карта развития', 'план']
}

def preprocess_question(question: str) -> str:
    """Удаляет вводные конструкции"""
    patterns = [
        r'^а если\s+', r'^что если\s+', r'^что будет если\s+',
        r'^можно ли\s+', r'^а что если\s+', r'^если я\s+',
        r'^а\s+', r'^ну\s+', r'^скажи\s+', r'^расскажи\s+', r'^объясни\s+'
    ]
    cleaned = question.lower()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    return cleaned.strip()

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    expanded = set(keywords)
    for word in keywords:
        for base, synonyms in SYNONYMS.items():
            if word == base or any(word == syn for syn in synonyms):
                expanded.update([base] + synonyms)
    return expanded

def load_knowledge_base(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    if not hasattr(lemmatize_word, 'cache'):
        lemmatize_word.cache = {}
    if word in lemmatize_word.cache:
        return lemmatize_word.cache[word]
    parsed = morph.parse(word)[0]
    lemma = parsed.normal_form
    lemmatize_word.cache[word] = lemma
    return lemma

def lemmatize_sentence(text: str) -> str:
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(word) for word in words if not is_stop_word(word) and len(word) > 2]
    return " ".join(lemmas)

def is_stop_word(word: str) -> bool:
    return word.lower() in RUSSIAN_STOPWORDS

def extract_keywords(text: str, use_synonyms: bool = True) -> set:
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    keywords = {lemmatize_word(word) for word in words if len(word) > 2 and not is_stop_word(word)}
    if use_synonyms:
        keywords = expand_with_synonyms(keywords)
    return keywords

def extract_entities(text: str) -> dict:
    entities = {
        'numbers': re.findall(r'\d+', text),
        'money': re.findall(r'\d+\s*(?:руб|р|рублей|долларов|usd|eur)', text, re.IGNORECASE),
        'timeframes': re.findall(r'\d+\s*(?:час|минут|дней|недел|месяц|год)', text, re.IGNORECASE)
    }
    return entities

def calculate_keyword_match_score(user_keywords: Set[str], item_keywords: Set[str], 
                                 user_question: str, original_keywords: List[str]) -> float:
    common_keywords = user_keywords.intersection(item_keywords)
    base_score = len(common_keywords) * 2
    
    question_lower = preprocess_text(user_question)
    phrase_bonus = 0
    for orig_keyword in original_keywords:
        keyword_lower = preprocess_text(orig_keyword)
        if keyword_lower in question_lower:
            phrase_bonus += len(keyword_lower.split()) * 3
            
    context_bonus = 0
    question_numbers = set(re.findall(r'\d+', user_question))
    keyword_numbers = set()
    for kw in original_keywords:
        keyword_numbers.update(re.findall(r'\d+', kw))
    if question_numbers and keyword_numbers and question_numbers.intersection(keyword_numbers):
        context_bonus += 5
    
    return base_score + phrase_bonus + context_bonus

# --- НОВАЯ ФУНКЦИЯ: Извлечение ссылок и создание кнопок ---
def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    """
    Находит ссылки в тексте, создает из них кнопки и удаляет их из текста.
    Возвращает очищенный текст и список кнопок.
    """
    buttons = []
    
    # Регулярка для поиска ссылок
    url_pattern = r'(https?://[^\s<]+|www\.[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for url in set(urls): # set убирает дубликаты
            # Пытаемся создать умное название кнопки
            label = "🔗 Ссылка"
            if "roadmap" in url.lower():
                label = "🗺 Дорожная карта"
            elif "Business-card" in url or "avick23.github.io" in url:
                label = "🌐 Сайт Алексея"
            elif "t.me" in url:
                label = "💬 Telegram"
            
            buttons.append([InlineKeyboardButton(label, url=url)])
        
        # Удаляем ссылки из текста, чтобы не дублировать
        clean_text = re.sub(url_pattern, '', text).strip()
        # Удаляем "мусорные" остатки (например, лишние скобки или пробелы перед точкой)
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        clean_text = re.sub(r'\(\s*\)', '', clean_text).strip()
        return clean_text, buttons
    
    return text, []

class KBIndex:
    # ... (класс KBIndex остается без изменений, он работает корректно) ...
    def __init__(self):
        self.items = []
        self.contexts = []
        self.tfidf_vectorizer = None
        self.tfidf_labeled_matrix = None
        self.raw_tfidf_vectorizer = None
        self.tfidf_raw_matrix = None
        self.last_update = 0
    
    def build_tfidf_index(self, contexts: List[str]):
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 3), max_features=3000
        )
        lemmatized_contexts = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized_contexts)
        
        self.raw_tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 2), max_features=2000
        )
        self.tfidf_raw_matrix = self.raw_tfidf_vectorizer.fit_transform(contexts)
    
    def keyword_search(self, user_question: str, top_k: int = 3) -> List[dict]:
        user_keywords = extract_keywords(user_question)
        if not user_keywords: return []
        scored_items = []
        for idx, item in enumerate(self.items):
            score = calculate_keyword_match_score(user_keywords, item["keywords"], user_question, item["original_keywords"])
            if score > 0:
                scored_items.append({"context": item["context"], "score": score, "index": idx})
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]
    
    def fulltext_search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.tfidf_vectorizer is None or self.tfidf_labeled_matrix is None: return []
        results = []
        try:
            query_lemma = lemmatize_sentence(query)
            query_vec = self.tfidf_vectorizer.transform([query_lemma])
            labeled_similarities = cosine_similarity(query_vec, self.tfidf_labeled_matrix)[0]
            
            raw_query_vec = self.raw_tfidf_vectorizer.transform([query])
            raw_similarities = cosine_similarity(raw_query_vec, self.tfidf_raw_matrix)[0]
            
            combined_similarities = 0.7 * labeled_similarities + 0.3 * raw_similarities
            top_indices = np.argsort(combined_similarities)[::-1][:top_k]
            
            for idx in top_indices:
                score = combined_similarities[idx]
                if score > 0.15:
                    results.append({"context": self.contexts[idx], "score": float(score), "index": int(idx)})
        except Exception as e:
            print(f"Ошибка TF-IDF: {e}")
        return results

def preprocess_knowledge_base(knowledge_base: list) -> KBIndex:
    kb_index = KBIndex()
    processed_items = []
    contexts = [item["context"] for item in knowledge_base]
    
    for i, item in enumerate(knowledge_base):
        processed_keywords = set()
        for keyword in item["keywords"]:
            for word in re.split(r'\s+', preprocess_text(keyword)):
                if len(word) > 2 and not is_stop_word(word):
                    processed_keywords.add(lemmatize_word(word))
        
        item_data = {
            "context": item["context"],
            "keywords": processed_keywords,
            "original_keywords": item["keywords"]
        }
        processed_items.append(item_data)
    
    kb_index.items = processed_items
    kb_index.contexts = contexts
    kb_index.build_tfidf_index(contexts)
    kb_index.last_update = time.time()
    return kb_index

def find_best_match(user_question: str, kb_index: KBIndex) -> str:
    cleaned_question = preprocess_question(user_question)
    entities = extract_entities(user_question)
    
    keyword_results = kb_index.keyword_search(cleaned_question, top_k=5)
    fulltext_results = kb_index.fulltext_search(cleaned_question, top_k=5)
    
    if not keyword_results and not fulltext_results:
        keyword_results = kb_index.keyword_search(user_question, top_k=5)
        fulltext_results = kb_index.fulltext_search(user_question, top_k=5)
    
    combined_results = {}
    for res in keyword_results:
        idx = res["index"]
        combined_results.setdefault(idx, 0)
        combined_results[idx] += res["score"] * 0.6
    
    for res in fulltext_results:
        idx = res["index"]
        combined_results.setdefault(idx, 0)
        combined_results[idx] += res["score"] * 50 * 0.4
    
    if combined_results:
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        best_idx, best_score = sorted_results[0]
        
        if best_score > 1.5:
            return kb_index.items[best_idx]["context"]
    
    if fulltext_results and fulltext_results[0]["score"] > 0.2:
        return fulltext_results[0]["context"]
    
    fallback_keywords = extract_keywords(cleaned_question, use_synonyms=False)
    if fallback_keywords:
        fallback_results = kb_index.keyword_search(" ".join(fallback_keywords), top_k=3)
        if fallback_results and fallback_results[0]["score"] > 0:
            return fallback_results[0]["context"]
            
    return "К сожалению, я не нашел ответа на ваш вопрос в своей базе знаний. Попробуйте задать вопрос другими словами или уточнить детали."

# Глобальные переменные
kb_index = None
user_contexts = {}

# --- ИЗМЕНЕНО: Добавили главное меню (клавиатуру внизу) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start с клавиатурой"""
    welcome_message = (
        "👋 Привет! Я Алексей, ваш цифровой помощник по обучению.\n\n"
        "Я знаю всё о моих методиках, дорожных картах и программе обучения.\n\n"
        "💡 Выберите действие в меню или задайте свой вопрос текстом:"
    )
    
    # Создаем клавиатуру с быстрыми действиями
    keyboard = [
        [KeyboardButton("🗓 Записаться на консультацию"), KeyboardButton("💰 Стоимость обучения")],
        [KeyboardButton("🗺 Дорожные карты"), KeyboardButton("🧠 О методе обучения")],
        [KeyboardButton("👨‍🏫 О преподавателе")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

# --- НОВОЕ: Команда /roadmaps ---
async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Быстрый доступ ко всем дорожным картам"""
    keyboard = [
        [InlineKeyboardButton("🐍 Python Roadmap", url="https://avick23.github.io/roadmap_python/")],
        [InlineKeyboardButton("⚡ Backend Roadmap", url="https://avick23.github.io/roadmap_backend/")],
        [InlineKeyboardButton("🐹 Golang Roadmap", url="https://avick23.github.io/roadmap_golang/")],
        [InlineKeyboardButton("🔧 DevOps Roadmap", url="https://avick23.github.io/roadmap_devops/")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🗺 <b>Мои дорожные карты (Roadmaps)</b>\n\n"
        "Это визуальные планы развития для разных направлений. "
        "Выберите интересующее вас направление:",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    user_data = {
        "user_id": user.id,
        "username": user.username or "Нет username",
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "timestamp": timestamp
    }
    
    consultations = []
    if os.path.exists(CONSULTATIONS_FILE):
        with open(CONSULTATIONS_FILE, "r", encoding="utf-8") as f:
            try:
                consultations = json.load(f)
            except json.JSONDecodeError:
                consultations = []
    
    consultations.append(user_data)
    
    with open(CONSULTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(consultations, f, ensure_ascii=False, indent=4)
    
    keyboard = [
        [InlineKeyboardButton("📅 Перейти к расписанию", url=CALENDAR_URL)],
        [InlineKeyboardButton("📱 Написать в Telegram", url="https://t.me/AVick23")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text="✅ <b>Ваша заявка успешно сохранена!</b>\n\n"
             "Вы можете:\n"
             "1. 🔗 Выбрать удобное время через Google Календарь\n"
             "2. 📱 Написать мне напрямую для согласования\n\n"
             "Я также свяжусь с вами в ближайшее время.",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def clear_list_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    with open(CONSULTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)
    
    await query.edit_message_text(text="✅ Список заявок успешно очищен!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_question = update.message.text.strip().lower()
    
    # Обработка команды "заявки" для администратора
    if user_id == ADMIN_USER_ID and user_question == "заявки":
        # ... (код без изменений) ...
        return
    
    # Инициализация контекста
    if user_id not in user_contexts:
        user_contexts[user_id] = {"last_answer": None}
    
    # Обработка коротких ответов
    short_answers = ['да', 'конечно', 'ага', 'угу', 'еще', 'больше', 'расскажи подробнее', 'как?', 'почему?']
    if user_question in short_answers:
        # ... (код без изменений) ...
        return

    # --- ОБРАБОТКА КНОПОК МЕНЮ (хардкод для удобства UX) ---
    if "записаться" in user_question and "консультаци" in user_question:
        keyboard = [
            [InlineKeyboardButton("📅 Перейти к расписанию", url=CALENDAR_URL)],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")]
        ]
        await update.message.reply_text(
            "Выберите удобный способ записи:", reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    if "стоимость" in user_question or "цена" in user_question:
        # Ищем ответ в базе знаний, если есть, иначе дефолт
        answer = find_best_match("стоимость обучения", kb_index)
        # Дальше логика стандартного ответа
    elif "дорожные карты" in user_question or "roadmap" in user_question:
        await roadmaps_command(update, context)
        return
    elif "о методе" in user_question:
        answer = find_best_match("метод выстраданного познания", kb_index)
    elif "о преподавателе" in user_question:
        answer = find_best_match("кто такой алексей", kb_index)
    else:
        # Стандартный поиск
        answer = find_best_match(update.message.text, kb_index)

    # Сохранение контекста
    clean_answer = answer.replace("[add_button]", "").strip()
    user_contexts[user_id]["last_answer"] = clean_answer
    
    # --- ИСПОЛЬЗУЕМ НОВУЮ ФУНКЦИЮ ДЛЯ ССЫЛОК ---
    display_text, url_buttons = extract_links_and_buttons(clean_answer)

    # Проверяем маркер добавления кнопки консультации
    if "[add_button]" in answer:
        url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])
    
    reply_markup = InlineKeyboardMarkup(url_buttons) if url_buttons else None
    
    # Если есть кнопки, отправляем с parse_mode HTML для жирного текста и т.д.
    if reply_markup:
        await update.message.reply_text(
            display_text, 
            reply_markup=reply_markup, 
            disable_web_page_preview=True, 
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(display_text)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"Произошла ошибка: {context.error}")
    if update and hasattr(update, 'message'):
        await update.message.reply_text("⚠️ Произошла ошибка. Попробуйте позже.")

def main() -> None:
    global kb_index
    
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("Токен бота не найден в .env файле. Укажите BOT_TOKEN=ваш_токен")
    
    try:
        kb = load_knowledge_base('main.json')
        kb_index = preprocess_knowledge_base(kb)
        print("✅ База знаний успешно загружена и обработана")
    except Exception as e:
        print(f"❌ Ошибка при загрузке базы знаний: {str(e)}")
        raise
    
    application = Application.builder().token(token).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command)) # Новая команда
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(consultation_callback, pattern="consultation"))
    application.add_handler(CallbackQueryHandler(clear_list_callback, pattern="clear_list"))
    application.add_error_handler(error_handler)
    
    print("🚀 Бот запущен и готов к работе!")
    application.run_polling()

if __name__ == "__main__":
    main()