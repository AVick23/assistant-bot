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
from datetime import datetime, timedelta
from collections import deque # Для безопасной очереди сообщений

# Загрузка переменных окружения
load_dotenv()

import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Импорт для нечеткого поиска
try:
    from thefuzz import process
    FUZZY_ENABLED = True
except ImportError:
    FUZZY_ENABLED = False
    print("⚠️ pip install thefuzz")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- КОНСТАНТЫ И ФАЙЛЫ ---
ADMIN_USER_ID = 1373472999
CONSULTATIONS_FILE = "consultations.json"
UNKNOWN_FILE = "unknown_questions.json"
FEEDBACK_FILE = "feedback.json"
CALENDAR_URL = "https://calendar.app.google/ThpteAc5uqhxqnUA9"
SITE_URL = "https://avick23.github.io/Business-card/"

ITEMS_PER_PAGE = 5

# КОНСТАНТЫ ПАМЯТИ
MAX_HISTORY_LENGTH = 5      # Хранить только последние 5 сообщений
INACTIVITY_LIMIT_HOURS = 24 # Удалять память через 24 часа неактивности

morph = pymorphy2.MorphAnalyzer()

# Стоп-слова (сокращено для компактности, вставьте свой полный список)
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'уже', 'всего', 'всё', 'быть', 'будет', 'сказал', 'этот', 'это', 'здесь', 'тот', 'там', 'где', 'который', 'которая', 'которые', 'их', 'этого', 'этой', 'этому', 'этим', 'эти', 'этих', 'ваш', 'ваша', 'ваше', 'вашего', 'вашей', 'какой', 'какая', 'какое', 'какие', 'какого', 'каком', 'какими', 'мы', 'наш', 'наша', 'наше', 'мой', 'моя', 'моё', 'мои', 'твой', 'твоя', 'твоё', 'твои', 'сам', 'сама', 'само', 'сами', 'тот', 'та', 'то', 'те', 'чей', 'чья', 'чьё', 'чьи', 'кто', 'что', 'где', 'куда', 'откуда', 'когда', 'почему', 'зачем', 'как', 'либо', 'нибудь', 'также', 'потому', 'чтобы', 'который', 'свой', 'своя', 'своё', 'свои', 'самый', 'самая', 'самое', 'самые', 'или', 'ну', 'эх', 'ах', 'ох', 'без', 'над', 'под', 'перед', 'после', 'между', 'через', 'чтобы', 'ради', 'для', 'до', 'после', 'около', 'возле', 'рядом', 'мимо', 'вокруг', 'против', 'за', 'надо', 'нужно', 'может', 'можно', 'должен', 'должна', 'должно', 'должны', 'хочу', 'хочешь', 'хочет', 'хотим', 'хотите', 'хотят', 'буду', 'будешь', 'будет', 'будем', 'будете', 'будут', 'хотя', 'если', 'пока', 'чтоб', 'зато', 'итак', 'также', 'тоже'
}

# Синонимы (сокращено, вставьте свой полный список)
SYNONYMS = {
    'стоимость': ['цена', 'тариф', 'плата', 'расценка', 'сколько стоит'],
    'курс': ['обучение', 'программа', 'тренинг'],
    'преподаватель': ['учитель', 'репетитор', 'тренер', 'лектор', 'алексей', 'avick23'],
    'метод': ['подход', 'техника', 'стратегия', 'выстраданного познания', 'система'],
    'бот': ['чат-бот', 'ассистент', 'помощник', 'прогресс', 'прогрессбот'],
    'python': ['питон', 'пайтон'],
    'roadmap': ['дорожная карта', 'карта развития', 'план']
}

# --- УТИЛИТЫ ---
def load_json(file_path):
    if not os.path.exists(file_path): return []
    try:
        with open(file_path, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)

# --- NLP ---
def preprocess_question(question: str) -> str:
    patterns = [r'^а если\s+', r'^что если\s+', r'^что будет если\s+', r'^можно ли\s+', r'^а что если\s+', r'^если я\s+', r'^а\s+', r'^ну\s+', r'^скажи\s+', r'^расскажи\s+', r'^объясни\s+']
    cleaned = question.lower()
    for pattern in patterns: cleaned = re.sub(pattern, '', cleaned)
    return cleaned.strip()

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    expanded = set(keywords)
    for word in keywords:
        for base, synonyms in SYNONYMS.items():
            if word == base or any(word == syn for syn in synonyms): expanded.update([base] + synonyms)
    return expanded

def load_knowledge_base(file_path: str) -> list:
    path = Path(file_path)
    if not path.exists(): raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)

def preprocess_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    if not hasattr(lemmatize_word, 'cache'): lemmatize_word.cache = {}
    if word in lemmatize_word.cache: return lemmatize_word.cache[word]
    parsed = morph.parse(word)[0]
    lemma = parsed.normal_form
    lemmatize_word.cache[word] = lemma
    return lemma

def lemmatize_sentence(text: str) -> str:
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(word) for word in words if not word in RUSSIAN_STOPWORDS and len(word) > 2]
    return " ".join(lemmas)

def extract_keywords(text: str, use_synonyms: bool = True) -> set:
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    keywords = {lemmatize_word(word) for word in words if len(word) > 2 and not word in RUSSIAN_STOPWORDS}
    if use_synonyms: keywords = expand_with_synonyms(keywords)
    return keywords

def calculate_keyword_match_score(user_keywords: Set[str], item_keywords: Set[str], user_question: str, original_keywords: List[str]) -> float:
    common_keywords = user_keywords.intersection(item_keywords)
    base_score = len(common_keywords) * 2
    question_lower = preprocess_text(user_question)
    phrase_bonus = 0
    for orig_keyword in original_keywords:
        keyword_lower = preprocess_text(orig_keyword)
        if keyword_lower in question_lower: phrase_bonus += len(keyword_lower.split()) * 3
    return base_score + phrase_bonus

def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "")
            clean_url = clean_url.strip('.,;:!?()"\'[]{}')
            if not clean_url: continue
            label = "🔗 Ссылка"
            if "roadmap" in clean_url.lower(): label = "🗺 Дорожная карта"
            elif "Business-card" in clean_url or "avick23.github.io" in clean_url: label = "🌐 Сайт Алексея"
            elif "t.me" in clean_url: label = "💬 Telegram"
            buttons.append([InlineKeyboardButton(label, url=clean_url)])
        clean_text = re.sub(url_pattern, '', text).strip()
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        clean_text = re.sub(r'\(\s*\)', '', clean_text).strip()
        return clean_text, buttons
    return text, []

# --- КЛАСС ИНДЕКСА ---
class KBIndex:
    def __init__(self):
        self.items = []
        self.contexts = []
        self.tfidf_vectorizer = None
        self.tfidf_labeled_matrix = None
        self.raw_tfidf_vectorizer = None
        self.tfidf_raw_matrix = None
        self.all_keywords_list = []
    
    def build_tfidf_index(self, contexts: List[str]):
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(RUSSIAN_STOPWORDS), ngram_range=(1, 3), max_features=3000)
        lemmatized_contexts = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized_contexts)
        self.raw_tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(RUSSIAN_STOPWORDS), ngram_range=(1, 2), max_features=2000)
        self.tfidf_raw_matrix = self.raw_tfidf_vectorizer.fit_transform(contexts)
        all_kw = set()
        for item in self.items: all_kw.update(item["original_keywords"])
        self.all_keywords_list = list(all_kw)
    
    def keyword_search(self, user_question: str, top_k: int = 3) -> List[dict]:
        user_keywords = extract_keywords(user_question)
        if not user_keywords: return []
        scored_items = []
        for idx, item in enumerate(self.items):
            score = calculate_keyword_match_score(user_keywords, item["keywords"], user_question, item["original_keywords"])
            if score > 0: scored_items.append({"context": item["context"], "score": score, "index": idx})
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]
    
    def fulltext_search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.tfidf_vectorizer is None or self.tfidf_labeled_matrix is None: return []
        try:
            query_lemma = lemmatize_sentence(query)
            query_vec = self.tfidf_vectorizer.transform([query_lemma])
            labeled_similarities = cosine_similarity(query_vec, self.tfidf_labeled_matrix)[0]
            raw_query_vec = self.raw_tfidf_vectorizer.transform([query])
            raw_similarities = cosine_similarity(raw_query_vec, self.tfidf_raw_matrix)[0]
            combined_similarities = 0.7 * labeled_similarities + 0.3 * raw_similarities
            top_indices = np.argsort(combined_similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = combined_similarities[idx]
                if score > 0.15: results.append({"context": self.contexts[idx], "score": float(score), "index": int(idx)})
            return results
        except: return []

def preprocess_knowledge_base(knowledge_base: list) -> KBIndex:
    kb_index = KBIndex()
    processed_items = []
    contexts = [item["context"] for item in knowledge_base]
    for i, item in enumerate(knowledge_base):
        processed_keywords = set()
        for keyword in item["keywords"]:
            for word in re.split(r'\s+', preprocess_text(keyword)):
                if len(word) > 2 and not word in RUSSIAN_STOPWORDS: processed_keywords.add(lemmatize_word(word))
        item_data = {"context": item["context"], "keywords": processed_keywords, "original_keywords": item["keywords"]}
        processed_items.append(item_data)
    kb_index.items = processed_items
    kb_index.contexts = contexts
    kb_index.build_tfidf_index(contexts)
    return kb_index

def search_knowledge_base(user_question: str, kb_index: KBIndex) -> Tuple[Optional[str], float, List[dict]]:
    cleaned_question = preprocess_question(user_question)
    keyword_results = kb_index.keyword_search(cleaned_question, top_k=5)
    fulltext_results = kb_index.fulltext_search(cleaned_question, top_k=5)
    if not keyword_results and not fulltext_results:
        keyword_results = kb_index.keyword_search(user_question, top_k=5)
        fulltext_results = kb_index.fulltext_search(user_question, top_k=5)
    
    combined_results = {}
    for res in keyword_results:
        combined_results.setdefault(res["index"], 0)
        combined_results[res["index"]] += res["score"] * 0.6
    for res in fulltext_results:
        combined_results.setdefault(res["index"], 0)
        combined_results[res["index"]] += res["score"] * 50 * 0.4
    
    if combined_results:
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for idx, score in sorted_results[:3]:
            topic_name = kb_index.items[idx]["original_keywords"][0] if kb_index.items[idx]["original_keywords"] else "Тема"
            candidates.append({"index": idx, "score": score, "topic": topic_name, "context": kb_index.items[idx]["context"]})
        best_idx, best_score = sorted_results[0]
        if best_score > 3.5: return kb_index.items[best_idx]["context"], best_score, candidates
        if best_score > 1.0: return kb_index.items[best_idx]["context"], best_score, candidates
    return None, 0.0, []

def get_fuzzy_suggestion(question: str, kb_index: KBIndex) -> Optional[str]:
    if not FUZZY_ENABLED or not kb_index.all_keywords_list: return None
    best_match, score = process.extractOne(question, kb_index.all_keywords_list)
    if score > 70: return best_match
    return None

kb_index = None
user_contexts = {} # Структура: {user_id: {"history": deque, "last_activity": datetime}}

# --- АДМИН-ПАНЕЛЬ ---
async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str, page: int = 0):
    query = update.callback_query
    if query: await query.answer()
    
    items = []
    title = ""
    empty_msg = ""
    clear_callback = ""
    
    if data_type == "consult":
        items = load_json(CONSULTATIONS_FILE)
        title = "📋 Заявки"
        empty_msg = "Заявок пока нет."
        clear_callback = "admin_clear_consult"
    elif data_type == "like":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "like"]
        title = "👍 Лайки"
        empty_msg = "Лайков пока нет."
        clear_callback = "admin_clear_like"
    elif data_type == "dislike":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "dislike"]
        title = "👎 Дизлайки"
        empty_msg = "Жалоб пока нет."
        clear_callback = "admin_clear_dislike"
    elif data_type == "unknown":
        items = load_json(UNKNOWN_FILE)
        title = "❓ Неизвестные вопросы"
        empty_msg = "Нет неизвестных вопросов."
        clear_callback = "admin_clear_unknown"

    total_items = len(items)
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
    if page < 0: page = 0
    if page >= total_pages: page = total_pages - 1
    
    text = f"<b>{title}</b> (Всего: {total_items})\n\n"
    
    if not items:
        text += f"<i>{empty_msg}</i>"
    else:
        start_idx = page * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        current_items = items[start_idx:end_idx]
        
        for i, item in enumerate(current_items, start=start_idx+1):
            if data_type == "consult":
                text += f"{i}. {item.get('first_name', '')} @{item.get('username', '')}\n   ⏰ {item.get('timestamp', '')}\n\n"
            elif data_type == "unknown":
                text += f"{i}. {item.get('question', '???')}\n\n"
            else:
                text += f"{i}. Q: {item.get('question', '???')[:30]}...\n\n"

    keyboard = []
    if total_pages > 1:
        nav_row = []
        if page > 0: nav_row.append(InlineKeyboardButton("◀️", callback_data=f"admin_page_{data_type}_{page-1}"))
        nav_row.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1: nav_row.append(InlineKeyboardButton("▶️", callback_data=f"admin_page_{data_type}_{page+1}"))
        keyboard.append(nav_row)
        
    if items: keyboard.append([InlineKeyboardButton("🗑 Очистить", callback_data=clear_callback)])
    if data_type != "consult": keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="admin_menu_main")])

    markup = InlineKeyboardMarkup(keyboard)
    
    if query:
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except: pass
    else:
        await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")

async def admin_clear_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    query = update.callback_query
    await query.answer()
    keyboard = [[InlineKeyboardButton("✅ Да", callback_data=f"admin_do_clear_{data_type}")], [InlineKeyboardButton("❌ Нет", callback_data=f"admin_page_{data_type}_0")]]
    await query.edit_message_text("⚠️ <b>Очистить список?</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    query = update.callback_query
    await query.answer()
    if data_type == "consult": save_json(CONSULTATIONS_FILE, [])
    elif data_type == "like" or data_type == "dislike":
        fb = load_json(FEEDBACK_FILE)
        save_json(FEEDBACK_FILE, [x for x in fb if x.get("type") != data_type])
    elif data_type == "unknown": save_json(UNKNOWN_FILE, [])
    await query.edit_message_text(f"✅ Очищено.", parse_mode="HTML")

# --- ОБРАБОТЧИКИ ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    # Инициализация памяти при старте
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=MAX_HISTORY_LENGTH), # Очередь с лимитом
            "last_activity": datetime.now()
        }
    
    text = "👋 Привет! Я Алексей, ваш цифровой помощник.\n\n💡 Меню:"
    keyboard = [
        [InlineKeyboardButton("🗓 Записаться", callback_data="menu_consult")],
        [InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"), InlineKeyboardButton("🗺 Дорожные карты", callback_data="menu_roadmaps")],
        [InlineKeyboardButton("🧠 О методе", callback_data="menu_method"), InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")]
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_admin_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    text = update.message.text.strip().lower()
    if user_id != ADMIN_USER_ID: return False
    if text in ["заявки", "заявка", "запись"]: await admin_show_list(update, context, "consult", 0); return True
    if text in ["отзыв", "отзывы", "лайки", "дизлайки"]:
        keyboard = [[InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"), InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")], [InlineKeyboardButton("❓ Неизвестные", callback_data="admin_page_unknown_0")]]
        await update.message.reply_text("<b>📊 Меню</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        return True
    return False

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    await query.answer()
    
    if data.startswith("admin_page_"): parts = data.split("_"); await admin_show_list(update, context, parts[2], int(parts[3])); return
    if data.startswith("admin_clear_"): await admin_clear_confirm(update, context, data.replace("admin_clear_", "")); return
    if data.startswith("admin_do_clear_"): await admin_do_clear(update, context, data.replace("admin_do_clear_", "")); return
    if data == "admin_menu_main": 
        keyboard = [[InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"), InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")], [InlineKeyboardButton("❓ Неизвестные", callback_data="admin_page_unknown_0")]]
        await query.edit_message_text("<b>📊 Меню</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
        return

    if data == "menu_consult": keyboard = [[InlineKeyboardButton("📅 Расписание", url=CALENDAR_URL)], [InlineKeyboardButton("📝 Заявка", callback_data="consultation")]]; await query.edit_message_text("Выберите:", reply_markup=InlineKeyboardMarkup(keyboard)); return
    if data == "menu_roadmaps": await roadmaps_command(update, context, edit_mode=True); return
        
    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {"menu_cost": "стоимость", "menu_method": "метод выстраданного познания", "menu_about": "кто такой алексей"}
        answer, _, candidates = search_knowledge_base(q_map[data], kb_index) if kb_index else ("Ошибка", 0, [])
        
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        
        if "[add_button]" in answer:
            url_buttons.append([InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")])
        
        ans_idx = 0
        if candidates: ans_idx = candidates[0]['index']
        else:
            for i, item in enumerate(kb_index.items):
                if item['context'] == answer: ans_idx = i; break
        
        url_buttons.append([InlineKeyboardButton("👍", callback_data=f"like_{ans_idx}"), InlineKeyboardButton("👎", callback_data=f"dislike_{ans_idx}")])

        await query.message.reply_text(display_text, reply_markup=InlineKeyboardMarkup(url_buttons), disable_web_page_preview=True, parse_mode="HTML")
        return

    if data.startswith("clarify_"):
        if data == "clarify_none": await query.edit_message_text("Хорошо."); return
        idx = int(data.split("_")[1])
        context_data = kb_index.items[idx]["context"]
        clean_text = context_data.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        if "[add_button]" in context_data: url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
        url_buttons.append([InlineKeyboardButton("👍", callback_data=f"like_{idx}"), InlineKeyboardButton("👎", callback_data=f"dislike_{idx}")])
        await query.edit_message_text(display_text, reply_markup=InlineKeyboardMarkup(url_buttons), parse_mode="HTML", disable_web_page_preview=True)
        return

    if data == "consultation": await consultation_callback(update, context); return
    if data.startswith("like_") or data.startswith("dislike_"): await feedback_callback(update, context); return

async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE, edit_mode: bool = False) -> None:
    keyboard = [
        [InlineKeyboardButton("🐍 Python", url="https://avick23.github.io/roadmap_python/")],
        [InlineKeyboardButton("⚡ Backend", url="https://avick23.github.io/roadmap_backend/")],
        [InlineKeyboardButton("🐹 Golang", url="https://avick23.github.io/roadmap_golang/")],
        [InlineKeyboardButton("🔧 DevOps", url="https://avick23.github.io/roadmap_devops/")]
    ]
    text = "🗺 <b>Дорожные карты</b>"
    if edit_mode: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    consultations = load_json(CONSULTATIONS_FILE)
    consultations.append({"user_id": user.id, "username": user.username or "Нет", "first_name": user.first_name or "", "last_name": user.last_name or "", "timestamp": timestamp})
    save_json(CONSULTATIONS_FILE, consultations)
    
    try: await context.bot.send_message(ADMIN_USER_ID, f"🔔 Заявка от @{user.username}", parse_mode="HTML")
    except: pass
    
    keyboard = [[InlineKeyboardButton("📅 Расписание", url=CALENDAR_URL)]]
    await query.edit_message_text("✅ <b>Заявка сохранена!</b>", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML")

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    user = query.from_user
    await query.answer()
    
    fb_type = "like" if "like_" in data else "dislike"
    idx = int(data.split("_")[1])
    
    answer = kb_index.items[idx]["context"] if kb_index else "???"
    # Берем последний вопрос из истории
    question = list(user_contexts.get(user.id, {}).get("history", []))[-1] if user_contexts.get(user.id) else "???"
    
    feedback_list = load_json(FEEDBACK_FILE)
    feedback_list.append({"type": fb_type, "question": question, "answer": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    save_json(FEEDBACK_FILE, feedback_list)
    
    await query.edit_message_reply_markup(None)
    if fb_type == "dislike":
        await query.message.reply_text("Спасибо за обратную связь!")
        try: await context.bot.send_message(ADMIN_USER_ID, f"👎 Дизлайк: {question}", parse_mode="HTML")
        except: pass

# --- ОСНОВНАЯ ЛОГИКА С КОНТЕКСТОМ ---

def get_contextual_question(user_id: int, current_question: str) -> str:
    """Добавляет контекст, если текущий вопрос короткий или уточняющий."""
    if user_id not in user_contexts:
        return current_question
    
    history = user_contexts[user_id]["history"]
    if not history:
        return current_question
    
    # Список слов-маркеров, которые требуют контекста
    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'а скидки', 'а рассрочка', 'а документ', 'а ты', 'а это']
    q_lower = current_question.lower()
    
    # Если вопрос короткий (< 20 символов) или содержит маркер контекста
    if len(q_lower) < 20 or any(marker in q_lower for marker in context_markers):
        # Берем последнее сообщение из истории
        last_msg = list(history)[-1]
        # Склеиваем. Например: "Стоимость обучения" + "а скидки есть?" -> "Стоимость обучения а скидки есть?"
        # Это поможет поиску найти ответ про скидки на обучение
        combined = f"{last_msg} {current_question}"
        return combined
    
    return current_question

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    user_question_lower = user_question.lower()
    
    # 1. Проверка админ-команд
    if await handle_admin_text(update, context): return

    # 2. Управление памятью (Garbage Collection)
    if user_id in user_contexts:
        # Проверяем время неактивности
        last_act = user_contexts[user_id].get("last_activity", datetime.now())
        if datetime.now() - last_act > timedelta(hours=INACTIVITY_LIMIT_HOURS):
            # Если пользователь молчал более 24 часов, сбрасываем его память
            print(f"Сброс памяти для {user_id} (неактивность)")
            del user_contexts[user_id]
    
    # Инициализация памяти
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=MAX_HISTORY_LENGTH),
            "last_activity": datetime.now()
        }
    
    # Обновляем время активности
    user_contexts[user_id]["last_activity"] = datetime.now()
    
    # 3. Добавляем вопрос в историю
    # deque автоматически удалит старый, если переполнен
    user_contexts[user_id]["history"].append(user_question)

    # 4. Формируем запрос с контекстом
    search_query = get_contextual_question(user_id, user_question)

    # 5. Поиск
    answer, score, candidates = search_knowledge_base(search_query, kb_index)
    final_answer = None
    
    if score > 3.5 and answer: final_answer = answer
    elif score > 1.5 and candidates:
        keyboard = [[InlineKeyboardButton(f"Ты про: {c['topic']}?", callback_data=f"clarify_{c['index']}")] for c in candidates]
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        await update.message.reply_text("Уточните:", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    elif FUZZY_ENABLED:
        suggestion = get_fuzzy_suggestion(user_question, kb_index)
        if suggestion:
            answer, score, candidates = search_knowledge_base(suggestion, kb_index)
            if score > 1.5: final_answer = answer
            if score < 3.5 and candidates:
                keyboard = [[InlineKeyboardButton(f"Может: {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]]
                await update.message.reply_text("Опечатка?", reply_markup=InlineKeyboardMarkup(keyboard))
                return

    if not final_answer:
        unk = load_json(UNKNOWN_FILE)
        unk.append({"question": user_question, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        save_json(UNKNOWN_FILE, unk)
        await update.message.reply_text("Не нашел ответа. /start")
        return

    clean_answer_for_memory = final_answer.replace("[add_button]", "").strip()
    user_contexts[user_id]["last_answer"] = clean_answer_for_memory
    
    display_text, url_buttons = extract_links_and_buttons(clean_answer_for_memory)
    
    if "[add_button]" in final_answer: url_buttons.append([InlineKeyboardButton("📝 Записаться", callback_data="consultation")])
    
    ans_idx = 0
    if candidates and candidates[0]['context'] == final_answer: ans_idx = candidates[0]['index']
    else:
        for i, item in enumerate(kb_index.items):
            if item['context'] == final_answer: ans_idx = i; break

    url_buttons.append([InlineKeyboardButton("👍", callback_data=f"like_{ans_idx}"), InlineKeyboardButton("👎", callback_data=f"dislike_{ans_idx}")])

    await update.message.reply_text(display_text, reply_markup=InlineKeyboardMarkup(url_buttons), disable_web_page_preview=True, parse_mode="HTML")

def main() -> None:
    global kb_index
    token = os.getenv("BOT_TOKEN")
    if not token: raise ValueError("Токен не найден")
    
    try:
        kb = load_knowledge_base('main.json')
        kb_index = preprocess_knowledge_base(kb)
        print("✅ База загружена")
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        return
    
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Бот запущен")
    application.run_polling()

if __name__ == "__main__":
    main()