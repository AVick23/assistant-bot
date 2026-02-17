import json
import re
import logging
import math
import os
import warnings
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import config

# Проверка нечеткого поиска
try:
    from thefuzz import process
    FUZZY_ENABLED = True
except ImportError:
    FUZZY_ENABLED = False
    print("⚠️ Библиотека thefuzz не установлена. Поиск опечаток отключен.")

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
logger = logging.getLogger(__name__)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
morph = pymorphy2.MorphAnalyzer()
user_contexts: Dict[int, dict] = {}
kb_index = None  # Инициализируется в main.py

# --- РАБОТА С JSON ---
def load_json(file_path: str) -> list:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def save_json(file_path: str, data: list) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error(f"Error saving {file_path}: {e}")

# --- NLP ФУНКЦИИ ---
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
        for base, synonyms in config.SYNONYMS.items():
            if word == base or any(word == syn for syn in synonyms):
                expanded.update([base] + synonyms)
    return expanded

def load_knowledge_base(file_path: str) -> list:
    if not os.path.exists(file_path):
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
    lemmas = [lemmatize_word(word) for word in words if word not in config.RUSSIAN_STOPWORDS and len(word) > 2]
    return " ".join(lemmas)

def extract_keywords(text: str, use_synonyms: bool = True) -> set:
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    keywords = {lemmatize_word(word) for word in words if len(word) > 2 and word not in config.RUSSIAN_STOPWORDS}
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
    return base_score + phrase_bonus

def extract_links_and_buttons(text: str) -> Tuple[str, List[List[InlineKeyboardButton]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "")
            clean_url = clean_url.strip('.,;:!?()"\'[]{}')
            if not clean_url:
                continue
            
            label = "🔗 Открыть ссылку"
            if "roadmap" in clean_url.lower():
                label = "🗺 Дорожная карта"
            elif "Business-card" in clean_url or "avick23.github.io" in clean_url:
                label = "👤 О преподавателе"
            elif "t.me" in clean_url:
                label = "💬 Telegram"
            elif "calendar" in clean_url.lower():
                label = "📅 Выбрать время"
            
            buttons.append([InlineKeyboardButton(label, url=clean_url)])
        
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
        self.all_keywords_list = []
    
    def build_tfidf_index(self, contexts: List[str]):
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words=list(config.RUSSIAN_STOPWORDS), 
            ngram_range=(1, 3), 
            max_features=3000
        )
        lemmatized_contexts = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized_contexts)
        
        self.raw_tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words=list(config.RUSSIAN_STOPWORDS), 
            ngram_range=(1, 2), 
            max_features=2000
        )
        self.tfidf_raw_matrix = self.raw_tfidf_vectorizer.fit_transform(contexts)
        
        all_kw = set()
        for item in self.items:
            all_kw.update(item["original_keywords"])
        self.all_keywords_list = list(all_kw)
    
    def keyword_search(self, user_question: str, top_k: int = 3) -> List[dict]:
        user_keywords = extract_keywords(user_question)
        if not user_keywords:
            return []
        
        scored_items = []
        for idx, item in enumerate(self.items):
            score = calculate_keyword_match_score(
                user_keywords, item["keywords"], user_question, item["original_keywords"]
            )
            if score > 0:
                scored_items.append({"context": item["context"], "score": score, "index": idx})
        
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]
    
    def fulltext_search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.tfidf_vectorizer is None or self.tfidf_labeled_matrix is None:
            return []
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
                if score > 0.15:
                    results.append({
                        "context": self.contexts[idx], 
                        "score": float(score), 
                        "index": int(idx)
                    })
            return results
        except Exception as e:
            logger.error(f"Fulltext search error: {e}")
            return []
    
    def is_valid_index(self, idx: int) -> bool:
        return 0 <= idx < len(self.items)

def preprocess_knowledge_base(knowledge_base: list) -> KBIndex:
    kb_index = KBIndex()
    processed_items = []
    contexts = [item["context"] for item in knowledge_base]
    
    for i, item in enumerate(knowledge_base):
        processed_keywords = set()
        for keyword in item["keywords"]:
            for word in re.split(r'\s+', preprocess_text(keyword)):
                if len(word) > 2 and word not in config.RUSSIAN_STOPWORDS:
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
            candidates.append({
                "index": idx, 
                "score": score, 
                "topic": topic_name, 
                "context": kb_index.items[idx]["context"]
            })
        
        best_idx, best_score = sorted_results[0]
        if best_score > 3.5:
            return kb_index.items[best_idx]["context"], best_score, candidates
        if best_score > 1.0:
            return kb_index.items[best_idx]["context"], best_score, candidates
    
    return None, 0.0, []

def get_fuzzy_suggestion(question: str, kb_index: KBIndex) -> Optional[str]:
    if not FUZZY_ENABLED or not kb_index.all_keywords_list:
        return None
    best_match, score = process.extractOne(question, kb_index.all_keywords_list)
    if score > 70:
        return best_match
    return None

# --- УПРАВЛЕНИЕ КОНТЕКСТОМ ---
def get_user_context(user_id: int) -> dict:
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=config.MAX_HISTORY_LENGTH),
            "last_activity": datetime.now(),
            "question_index_map": {},
        }
    return user_contexts[user_id]

def update_user_activity(user_id: int) -> None:
    ctx = get_user_context(user_id)
    ctx["last_activity"] = datetime.now()

def cleanup_inactive_users() -> None:
    now = datetime.now()
    to_delete = [
        uid for uid, ctx in user_contexts.items()
        if now - ctx.get("last_activity", now) > timedelta(hours=config.INACTIVITY_LIMIT_HOURS)
    ]
    for uid in to_delete:
        del user_contexts[uid]

def save_question_for_answer(user_id: int, answer_index: int, question: str) -> None:
    ctx = get_user_context(user_id)
    ctx["question_index_map"][answer_index] = question

def get_question_for_answer(user_id: int, answer_index: int) -> str:
    ctx = get_user_context(user_id)
    return ctx.get("question_index_map", {}).get(answer_index, "???")

def get_contextual_question(user_id: int, current_question: str) -> str:
    ctx = get_user_context(user_id)
    history = ctx.get("history", [])
    
    if not history:
        return current_question
    
    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'а скидки', 'а рассрочка', 'а документ']
    q_lower = current_question.lower()
    
    if len(q_lower) < 20 or any(marker in q_lower for marker in context_markers):
        last_msg = list(history)[-1] if history else ""
        return f"{last_msg} {current_question}"
    
    return current_question

# --- КЛАВИАТУРЫ ---
class AppleKeyboards:
    """Централизованное создание клавиатур с учетом роли (Админ/Юзер)"""
    
    @staticmethod
    def main_menu(user_id: int) -> InlineKeyboardMarkup:
        is_admin = (user_id == config.ADMIN_USER_ID)
        
        keyboard = [
            [InlineKeyboardButton("🗓 Записаться на консультацию", callback_data="menu_consult")],
            [
                InlineKeyboardButton("💰 Стоимость", callback_data="menu_cost"),
                InlineKeyboardButton("🗺 Карты обучения", callback_data="menu_roadmaps")
            ],
            [
                InlineKeyboardButton("🧠 О методе", callback_data="menu_method"),
                InlineKeyboardButton("👨‍🏫 О преподавателе", callback_data="menu_about")
            ],
        ]
        
        # Админ-кнопка внизу главного меню
        if is_admin:
            keyboard.append([InlineKeyboardButton("🔐 Админ-панель", callback_data="admin_menu_main")])
            
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def feedback_buttons(user_id: int, answer_index: int) -> List[List[InlineKeyboardButton]]:
        """Кнопки обратной связи. АДМИН их не видит."""
        if user_id == config.ADMIN_USER_ID:
            return []
            
        return [
            [
                InlineKeyboardButton("👍 Полезно", callback_data=f"like_{answer_index}"),
                InlineKeyboardButton("👎 Не помогло", callback_data=f"dislike_{answer_index}")
            ]
        ]

    @staticmethod
    def consult_menu(user_id: int) -> InlineKeyboardMarkup:
        is_admin = (user_id == config.ADMIN_USER_ID)
        keyboard = [
            [InlineKeyboardButton("📅 Выбрать время в календаре", url=config.CALENDAR_URL)],
        ]
        
        if not is_admin:
            keyboard.append([InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")])
            
        keyboard.append([InlineKeyboardButton("◀️ Назад", callback_data="menu_main")])
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def roadmaps_menu() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🐍 Python", url="https://avick23.github.io/roadmap_python/")],
            [InlineKeyboardButton("⚡ Backend", url="https://avick23.github.io/roadmap_backend/")],
            [InlineKeyboardButton("🐹 Golang", url="https://avick23.github.io/roadmap_golang/")],
            [InlineKeyboardButton("🔧 DevOps", url="https://avick23.github.io/roadmap_devops/")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def back_button(callback_data: str = "menu_main") -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=callback_data)]])

    @staticmethod
    def admin_panel_main() -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("📊 Статистика (Лайки/Дизлайки)", callback_data="admin_stats")],
            [InlineKeyboardButton("❓ Неизвестные вопросы", callback_data="admin_page_unknown_0")],
            [InlineKeyboardButton("📋 Заявки на консультацию", callback_data="admin_page_consult_0")],
            [InlineKeyboardButton("◀️ Выйти в главное меню", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)

# --- Уведомления для Админа ---
async def notify_admin(context, text: str):
    """Отправка уведомления админу"""
    try:
        await context.bot.send_message(config.ADMIN_USER_ID, text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Failed to notify admin: {e}")