import json
import re
import numpy as np
import warnings
from typing import Dict, List, Set, Optional, Tuple, Any
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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Импорт для нечеткого поиска (установите: pip install thefuzz)
try:
    from thefuzz import process
    FUZZY_ENABLED = True
except ImportError:
    FUZZY_ENABLED = False
    print("⚠️ Библиотека thefuzz не установлена. Поиск опечаток отключен.")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- КОНСТАНТЫ ---
ADMIN_USER_ID = 1373472999
CONSULTATIONS_FILE = "consultations.json"
UNKNOWN_FILE = "unknown_questions.json"
FEEDBACK_FILE = "feedback.json"
CALENDAR_URL = "https://calendar.app.google/ThpteAc5uqhxqnUA9"
SITE_URL = "https://avick23.github.io/Business-card/"

morph = pymorphy2.MorphAnalyzer()

# Стоп-слова и синонимы (оставлены без изменений)
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет',
    'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'уже', 'всего', 'всё',
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

SYNONYMS = {
    'стоимость': ['цена', 'тариф', 'плата', 'расценка', 'сколько стоит'],
    'курс': ['обучение', 'программа', 'тренинг'],
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ NLP ---

def preprocess_question(question: str) -> str:
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

def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+|www\.[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for url in set(urls):
            label = "🔗 Ссылка"
            if "roadmap" in url.lower():
                label = "🗺 Дорожная карта"
            elif "Business-card" in url or "avick23.github.io" in url:
                label = "🌐 Сайт Алексея"
            elif "t.me" in url:
                label = "💬 Telegram"
            buttons.append([InlineKeyboardButton(label, url=url)])
        
        clean_text = re.sub(url_pattern, '', text).strip()
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        clean_text = re.sub(r'\(\s*\)', '', clean_text).strip()
        return clean_text, buttons
    return text, []

# --- КЛАСС ИНДЕКСА БАЗЫ ЗНАНИЙ ---

class KBIndex:
    def __init__(self):
        self.items = []
        self.contexts = []
        self.tfidf_vectorizer = None
        self.tfidf_labeled_matrix = None
        self.raw_tfidf_vectorizer = None
        self.tfidf_raw_matrix = None
        self.all_keywords_list = [] # Для нечеткого поиска
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
        
        # Собираем все ключевые слова для Fuzzy поиска
        all_kw = set()
        for item in self.items:
            all_kw.update(item["original_keywords"])
        self.all_keywords_list = list(all_kw)
    
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

def search_knowledge_base(user_question: str, kb_index: KBIndex) -> Tuple[Optional[str], float, List[dict]]:
    """
    Возвращает: (лучший ответ, оценка, список кандидатов)
    Оценка используется для логики уточнения.
    """
    cleaned_question = preprocess_question(user_question)
    
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
        
        # Формируем список кандидатов (нужно для уточнения)
        candidates = []
        for idx, score in sorted_results[:3]:
             # Берем первое ключевое слово как "тему" для кнопки
            topic_name = kb_index.items[idx]["original_keywords"][0] if kb_index.items[idx]["original_keywords"] else "Тема"
            candidates.append({
                "index": idx,
                "score": score,
                "topic": topic_name,
                "context": kb_index.items[idx]["context"]
            })
            
        best_idx, best_score = sorted_results[0]
        
        # Если оценка очень высокая - отвечаем сразу
        if best_score > 3.5:
            return kb_index.items[best_idx]["context"], best_score, candidates
            
        # Если средняя - возможно понадобится уточнение
        if best_score > 1.0:
             return kb_index.items[best_idx]["context"], best_score, candidates
    
    return None, 0.0, []

def get_fuzzy_suggestion(question: str, kb_index: KBIndex) -> Optional[str]:
    if not FUZZY_ENABLED or not kb_index.all_keywords_list:
        return None
        
    # Ищем совпадение с порогом 70%
    best_match, score = process.extractOne(question, kb_index.all_keywords_list)
    if score > 70:
        return best_match
    return None

# Глобальные переменные
kb_index = None
user_contexts = {}

# --- ФУНКЦИЯ ЛОГИРОВАНИЯ ---

def log_unknown_question(question: str):
    """Сохраняет вопросы, на которые бот не нашел ответа"""
    data = []
    if os.path.exists(UNKNOWN_FILE):
        try:
            with open(UNKNOWN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except: pass
    
    data.append({
        "question": question,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with open(UNKNOWN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- ОБРАБОТЧИКИ TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    # Сохраняем пользователя, если его нет
    if user_id not in user_contexts:
        user_contexts[user_id] = {"last_answer": None, "last_raw_question": None}

    welcome_message = (
        "👋 Привет! Я Алексей, ваш цифровой помощник по обучению.\n\n"
        "Я знаю всё о моих методиках, дорожных картах и программе обучения.\n\n"
        "💡 Выберите действие или задайте вопрос текстом:"
    )
    
    # Главное меню (Inline кнопки)
    keyboard = [
        [InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="menu_consult")],
        [InlineKeyboardButton("💰 Стоимость обучения", callback_data="menu_cost")],
        [InlineKeyboardButton("🗺 Дорожные карты", callback_data="menu_roadmaps")],
        [InlineKeyboardButton("🧠 О методе обучения", callback_data="menu_method")],
        [InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

# --- ОБРАБОТЧИК INLINE КНОПОК (Menu & Feedback & Clarify) ---

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    # Обработка главного меню
    if data == "menu_consult":
        keyboard = [
            [InlineKeyboardButton("📅 Перейти к расписанию", url=CALENDAR_URL)],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")]
        ]
        await query.edit_message_text("Выберите удобный способ записи:", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    
    if data == "menu_roadmaps":
        await roadmaps_command(update, context, edit_mode=True)
        return
        
    if data == "menu_cost":
        answer = find_best_match("стоимость обучения", kb_index) if kb_index else "База знаний недоступна"
        # Отправляем как новое сообщение, так как это выбор из меню
        await query.message.reply_text(answer)
        return

    if data == "menu_method":
        answer = find_best_match("метод выстраданного познания", kb_index) if kb_index else "База знаний недоступна"
        await query.message.reply_text(answer)
        return
        
    if data == "menu_about":
        answer = find_best_match("кто такой алексей", kb_index) if kb_index else "База знаний недоступна"
        await query.message.reply_text(answer)
        return

    # Обработка уточнения (Clarification)
    if data.startswith("clarify_"):
        idx = int(data.split("_")[1])
        context_data = kb_index.items[idx]["context"]
        # Убираем маркер добавления кнопки записи, если он есть в контексте
        clean_text = context_data.replace("[add_button]", "").strip()
        
        # Добавляем кнопки ссылок и лайки
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        # Добавляем кнопку записи
        if "[add_button]" in context_data:
            url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])
        
        # Добавляем feedback
        url_buttons.append([
            InlineKeyboardButton("👍", callback_data=f"like_{idx}"),
            InlineKeyboardButton("👎", callback_data=f"dislike_{idx}")
        ])
        
        await query.edit_message_text(display_text, reply_markup=InlineKeyboardMarkup(url_buttons), parse_mode="HTML", disable_web_page_preview=True)
        return

    # Обработка записи
    if data == "consultation":
        await consultation_callback(update, context)
        return

    # Обработка лайков/дизлайков
    if data.startswith("like_") or data.startswith("dislike_"):
        await feedback_callback(update, context)
        return

async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE, edit_mode: bool = False) -> None:
    keyboard = [
        [InlineKeyboardButton("🐍 Python Roadmap", url="https://avick23.github.io/roadmap_python/")],
        [InlineKeyboardButton("⚡ Backend Roadmap", url="https://avick23.github.io/roadmap_backend/")],
        [InlineKeyboardButton("🐹 Golang Roadmap", url="https://avick23.github.io/roadmap_golang/")],
        [InlineKeyboardButton("🔧 DevOps Roadmap", url="https://avick23.github.io/roadmap_devops/")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = ("🗺 <b>Мои дорожные карты (Roadmaps)</b>\n\n"
            "Это визуальные планы развития для разных направлений. "
            "Выберите интересующее вас направление:")
    
    if edit_mode:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode="HTML")

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
    
    # Сохранение заявки
    consultations = []
    if os.path.exists(CONSULTATIONS_FILE):
        try: consultations = json.load(open(CONSULTATIONS_FILE, "r", encoding="utf-8"))
        except: pass
    consultations.append(user_data)
    json.dump(consultations, open(CONSULTATIONS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    
    # Уведомление админа
    try:
        admin_msg = (f"🔔 <b>Новая заявка!</b>\n\n👤 <b>Имя:</b> {user.first_name or ''} {user.last_name or ''}\n"
                     f"🆔 <b>Username:</b> @{user.username if user.username else 'не указан'}\n"
                     f"⏰ <b>Время:</b> {timestamp}")
        admin_kb = []
        if user.username:
            admin_kb.append([InlineKeyboardButton("💬 Написать", url=f"tg://resolve?domain={user.username}")])
        await context.bot.send_message(ADMIN_USER_ID, admin_msg, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(admin_kb) if admin_kb else None)
    except Exception as e:
        print(f"Ошибка уведомления админа: {e}")
    
    # Ответ пользователю
    keyboard = [
        [InlineKeyboardButton("📅 Перейти к расписанию", url=CALENDAR_URL)],
        [InlineKeyboardButton("📱 Написать в Telegram", url="https://t.me/AVick23")]
    ]
    await query.edit_message_text("✅ <b>Ваша заявка успешно сохранена!</b>\n\nЯ свяжусь с вами в ближайшее время.", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user = query.from_user
    
    # Если это дизлайк, логируем и уведомляем админа
    if "dislike" in data:
        idx = int(data.split("_")[1])
        # Получаем контекст ответа
        bad_context = kb_index.items[idx]["context"] if kb_index else "Неизвестный ответ"
        original_question = user_contexts.get(user.id, {}).get("last_raw_question", "Неизвестный вопрос")
        
        # Уведомление админу
        try:
            msg = (f"👎 <b>Плохой ответ!</b>\n\n"
                   f"❓ <b>Вопрос:</b> {original_question}\n"
                   f"💬 <b>Ответ бота:</b> {bad_context[:100]}...")
            await context.bot.send_message(ADMIN_USER_ID, msg, parse_mode="HTML")
        except: pass
        
        # Сохраняем в файл
        fb_data = []
        if os.path.exists(FEEDBACK_FILE):
            try: fb_data = json.load(open(FEEDBACK_FILE, "r", encoding="utf-8"))
            except: pass
        fb_data.append({"question": original_question, "bad_answer": bad_context, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        json.dump(fb_data, open(FEEDBACK_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        
        await query.edit_message_reply_markup(None) # Убираем кнопки
        await query.message.reply_text("Спасибо за обратную связь! Я учту это для улучшения ответов.")
    
    elif "like" in data:
        await query.edit_message_reply_markup(None)
        # Можно просто убрать кнопки или ответить тихо

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    user_question_lower = user_question.lower()
    
    # Инициализация контекста
    if user_id not in user_contexts:
        user_contexts[user_id] = {"last_answer": None, "last_raw_question": None}
    
    # Админ-команда
    if user_id == ADMIN_USER_ID and user_question_lower == "заявки":
        # Логика просмотра заявок (упрощенно, без изменений)
        if not os.path.exists(CONSULTATIONS_FILE):
            await update.message.reply_text("📋 Список заявок пуст.")
            return
        # ... (код просмотра заявок можно оставить старый или упростить для краткости)
        await update.message.reply_text("📋 Проверьте файл consultations.json на сервере.")
        return

    # Контекст: сохраняем текущий вопрос
    user_contexts[user_id]["last_raw_question"] = user_question

    # 1. Пробуем стандартный поиск
    answer, score, candidates = search_knowledge_base(user_question, kb_index)
    
    # 2. Логика ответа
    final_answer = None
    candidates_keyboard = []
    
    # А. Уверенный ответ
    if score > 3.5 and answer:
        final_answer = answer
    # Б. Средняя уверенность -> Уточнение
    elif score > 1.5 and candidates:
        # Предлагаем уточнить
        keyboard = []
        for cand in candidates:
            keyboard.append([InlineKeyboardButton(f"Ты про: {cand['topic']}?", callback_data=f"clarify_{cand['index']}")])
        keyboard.append([InlineKeyboardButton("❌ Это не то", callback_data="clarify_none")])
        
        await update.message.reply_text(
            "Я не совсем уверен, что вы имели в виду. Вы спрашивали про:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
    # В. Низкая уверенность -> Пробуем Fuzzy Search
    elif FUZZY_ENABLED:
        suggestion = get_fuzzy_suggestion(user_question, kb_index)
        if suggestion:
            # Нашли опечатку -> перезапускаем поиск
            answer, score, candidates = search_knowledge_base(suggestion, kb_index)
            if score > 1.5:
                final_answer = answer
                # Если все равно средне, можно снова уточнить, но пока просто ответим
                # Или предложим один вариант
                if score < 3.5 and candidates:
                    keyboard = [[InlineKeyboardButton(f"Может вы имели в виду: {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]]
                    await update.message.reply_text("Возможно, вы опечатались?", reply_markup=InlineKeyboardMarkup(keyboard))
                    return

    # Г. Ничего не нашли -> Логирование
    if not final_answer:
        log_unknown_question(user_question)
        await update.message.reply_text(
            "К сожалению, я не нашел ответа в своей базе знаний. "
            "Я сохранил ваш вопрос, чтобы стать умнее в будущем.\n\n"
            "Попробуйте сформулировать иначе или используйте меню /start."
        )
        return

    # Отправка финального ответа
    # Очистка от маркеров и извлечение ссылок
    clean_answer_for_memory = final_answer.replace("[add_button]", "").strip()
    user_contexts[user_id]["last_answer"] = clean_answer_for_memory
    
    display_text, url_buttons = extract_links_and_buttons(clean_answer_for_memory)
    
    # Кнопки действий
    if "[add_button]" in final_answer:
        url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])
    
    # Кнопки оценки
    # Нам нужен индекс ответа для лайка. Найдем его.
    ans_idx = 0
    if candidates and candidates[0]['context'] == final_answer:
        ans_idx = candidates[0]['index']
    else:
        # Найдем индекс перебором (не оптимально, но надежно для фидбека)
        for i, item in enumerate(kb_index.items):
            if item['context'] == final_answer:
                ans_idx = i
                break

    url_buttons.append([
        InlineKeyboardButton("👍", callback_data=f"like_{ans_idx}"),
        InlineKeyboardButton("👎", callback_data=f"dislike_{ans_idx}")
    ])

    reply_markup = InlineKeyboardMarkup(url_buttons)
    
    await update.message.reply_text(
        display_text, 
        reply_markup=reply_markup, 
        disable_web_page_preview=True, 
        parse_mode="HTML"
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"Ошибка: {context.error}")

# --- ЗАПУСК ---

def main() -> None:
    global kb_index
    
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("Токен не найден в .env")
    
    try:
        kb = load_knowledge_base('main.json')
        kb_index = preprocess_knowledge_base(kb)
        print("✅ База знаний загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки KB: {str(e)}")
        return
    
    application = Application.builder().token(token).build()
    
    # Хендлеры
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command))
    
    # Главный обработчик кнопок (Menu, Consultation, Feedback, Clarification)
    application.add_handler(CallbackQueryHandler(menu_callback))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    print("🚀 Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()