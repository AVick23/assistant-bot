"""
🎯 Прогресс Бот v2.1
- Убраны текстовые команды админа (заявки/отзывы).
- Добавлена кнопка "Панель управления" в главное меню (видна только админу).
- UX в стиле Apple: чистота, минимализм, контекстность.
"""

import json
import re
import numpy as np
import warnings
import logging
import traceback
from typing import Dict, List, Set, Optional, Tuple, Any
import math
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import deque

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
    print("⚠️ Библиотека thefuzz не установлена. pip install thefuzz")

# --- ЛОГИРОВАНИЕ ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- КОНСТАНТЫ ---
ADMIN_USER_ID = 1373472999  # Убедитесь, что здесь ваш ID
CONSULTATIONS_FILE = "consultations.json"
UNKNOWN_FILE = "unknown_questions.json"
FEEDBACK_FILE = "feedback.json"
CALENDAR_URL = "https://calendar.app.google/ThpteAc5uqhxqnUA9"
SITE_URL = "https://avick23.github.io/Business-card/"

ITEMS_PER_PAGE = 5
MAX_HISTORY_LENGTH = 5
INACTIVITY_LIMIT_HOURS = 24

morph = pymorphy2.MorphAnalyzer()

# Стоп-слова
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'уже', 'всего', 'всё', 'быть', 'будет', 'сказал', 'этот', 'это', 'здесь', 'тот', 'там', 'где', 'который', 'которая', 'которые', 'их', 'этого', 'этой', 'этому', 'этим', 'эти', 'этих', 'ваш', 'ваша', 'ваше', 'вашего', 'вашей', 'какой', 'какая', 'какое', 'какие', 'какого', 'каком', 'какими', 'мы', 'наш', 'наша', 'наше', 'мой', 'моя', 'моё', 'мои', 'твой', 'твоя', 'твоё', 'твои', 'сам', 'сама', 'само', 'сами', 'тот', 'та', 'то', 'те', 'чей', 'чья', 'чьё', 'чьи', 'кто', 'что', 'где', 'куда', 'откуда', 'когда', 'почему', 'зачем', 'как', 'либо', 'нибудь', 'также', 'потому', 'чтобы', 'который', 'свой', 'своя', 'своё', 'свои', 'самый', 'самая', 'самое', 'самые', 'или', 'ну', 'эх', 'ах', 'ох', 'без', 'над', 'под', 'перед', 'после', 'между', 'через', 'чтобы', 'ради', 'для', 'до', 'после', 'около', 'возле', 'рядом', 'мимо', 'вокруг', 'против', 'за', 'надо', 'нужно', 'может', 'можно', 'должен', 'должна', 'должно', 'должны', 'хочу', 'хочешь', 'хочет', 'хотим', 'хотите', 'хотят', 'буду', 'будешь', 'будет', 'будем', 'будете', 'будут', 'хотя', 'если', 'пока', 'чтоб', 'зато', 'итак', 'также', 'тоже'
}

# Синонимы
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


# ============================================================
# 🎨 APPLE-STYLE UX: Тексты и сообщения
# ============================================================

class AppleStyleMessages:
    """Централизованное хранение всех сообщений в стиле Apple"""
    
    WELCOME = """👋 Привет!

Я — ваш персональный помощник по обучению программированию.

💡 Просто напишите вопрос, и я помогу найти ответ.

👇 Или выберите тему ниже:"""

    WELCOME_RETURNING = """👋 С возвращением!

Чем могу помочь сегодня?"""

    HELP = """📚 <b>Как пользоваться ботом</b>

Просто пишите вопросы на естественном языке — я пойму.

<b>Примеры вопросов:</b>
• «Сколько стоит обучение?»
• «Расскажи о преподавателе»
• «Как записаться на консультацию?»

<b>Возможности:</b>
• Поиск ответов в базе знаний
• Запись на консультацию
• Дорожные карты обучения

<i>Я запоминаю контекст беседы, поэтому можно задавать уточняющие вопросы.</i>"""

    NOT_FOUND = """🤔 <b>Пока не знаю ответа</b>

Но я сохранил ваш вопрос — скоро научусь на него отвечать.

<b>Попробуйте:</b>
• Переформулировать вопрос
• Выбрать тему в меню /start
• Написать /help"""

    CONSULTATION_SUCCESS = """✅ <b>Заявка отправлена</b>

Алексей свяжется с вами в ближайшее время.

📅 А пока можете выбрать удобное время в календаре:"""

    FEEDBACK_THANKS = """💚 Спасибо за оценку!

Ваше мнение помогает становиться лучше."""

    FEEDBACK_DISLIKE = """📝 Спасибо за обратную связь

Ваш отзыв отправлен разработчику. Мы постараемся улучшить ответы."""

    CLARIFY_PROMPT = """🤔 Уточните, пожалуйста:"""

    FUZZY_SUGGESTION = """💡 Возможно, вы имели в виду:"""


# ============================================================
# 🛠 УТИЛИТЫ
# ============================================================

def load_json(file_path: str) -> list:
    """Безопасная загрузка JSON"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def save_json(file_path: str, data: list) -> None:
    """Безопасное сохранение JSON"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error(f"Error saving {file_path}: {e}")


# ============================================================
# 🧠 NLP ФУНКЦИИ
# ============================================================

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
    lemmas = [lemmatize_word(word) for word in words if word not in RUSSIAN_STOPWORDS and len(word) > 2]
    return " ".join(lemmas)


def extract_keywords(text: str, use_synonyms: bool = True) -> set:
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    keywords = {lemmatize_word(word) for word in words if len(word) > 2 and word not in RUSSIAN_STOPWORDS}
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
    """Извлекает ссылки и создаёт красивые кнопки"""
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "")
            clean_url = clean_url.strip('.,;:!?()"\'[]{}')
            if not clean_url:
                continue
            
            # 🎨 Apple-style названия кнопок
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


# ============================================================
# 📚 КЛАСС ИНДЕКСА БАЗЫ ЗНАНИЙ
# ============================================================

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
            stop_words=list(RUSSIAN_STOPWORDS), 
            ngram_range=(1, 3), 
            max_features=3000
        )
        lemmatized_contexts = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized_contexts)
        
        self.raw_tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words=list(RUSSIAN_STOPWORDS), 
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
        """Проверка валидности индекса"""
        return 0 <= idx < len(self.items)


def preprocess_knowledge_base(knowledge_base: list) -> KBIndex:
    kb_index = KBIndex()
    processed_items = []
    contexts = [item["context"] for item in knowledge_base]
    
    for i, item in enumerate(knowledge_base):
        processed_keywords = set()
        for keyword in item["keywords"]:
            for word in re.split(r'\s+', preprocess_text(keyword)):
                if len(word) > 2 and word not in RUSSIAN_STOPWORDS:
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


# ============================================================
# 🌐 ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================

kb_index: Optional[KBIndex] = None
user_contexts: Dict[int, dict] = {}  # {user_id: {"history": deque, "last_activity": datetime, ...}}


# ============================================================
# 🎨 APPLE-STYLE КЛАВИАТУРЫ
# ============================================================

class AppleKeyboards:
    """Централизованное создание клавиатур в стиле Apple"""
    
    @staticmethod
    def main_menu(is_returning: bool = False, is_admin: bool = False) -> InlineKeyboardMarkup:
        """Главное меню - чистое и понятное"""
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
        
        # 🎯 Специальная кнопка для админа
        if is_admin:
            keyboard.append([InlineKeyboardButton("⚙️ Панель управления", callback_data="admin_panel")])
            
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def admin_panel() -> InlineKeyboardMarkup:
        """Меню администратора - минимализм и функциональность"""
        keyboard = [
            [InlineKeyboardButton("📋 Заявки на консультацию", callback_data="admin_page_consult_0")],
            [
                InlineKeyboardButton("👍 Лайки", callback_data="admin_page_like_0"),
                InlineKeyboardButton("👎 Дизлайки", callback_data="admin_page_dislike_0")
            ],
            [InlineKeyboardButton("❓ Неизвестные вопросы", callback_data="admin_page_unknown_0")],
            [InlineKeyboardButton("◀️ Главное меню", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def feedback_buttons(answer_index: int) -> List[List[InlineKeyboardButton]]:
        """Кнопки обратной связи - интуитивные"""
        return [
            [
                InlineKeyboardButton("👍 Полезно", callback_data=f"like_{answer_index}"),
                InlineKeyboardButton("👎 Не помогло", callback_data=f"dislike_{answer_index}")
            ]
        ]
    
    @staticmethod
    def consult_menu() -> InlineKeyboardMarkup:
        """Меню консультации"""
        keyboard = [
            [InlineKeyboardButton("📅 Выбрать время в календаре", url=CALENDAR_URL)],
            [InlineKeyboardButton("📝 Оставить заявку", callback_data="consultation")],
            [InlineKeyboardButton("◀️ Назад", callback_data="menu_main")]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    @staticmethod
    def roadmaps_menu() -> InlineKeyboardMarkup:
        """Меню дорожных карт"""
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
        """Кнопка назад"""
        return InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=callback_data)]])


# ============================================================
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_user_context(user_id: int) -> dict:
    """Получение или создание контекста пользователя"""
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=MAX_HISTORY_LENGTH),
            "last_activity": datetime.now(),
            "question_index_map": {},
        }
    return user_contexts[user_id]


def update_user_activity(user_id: int) -> None:
    """Обновление активности пользователя"""
    ctx = get_user_context(user_id)
    ctx["last_activity"] = datetime.now()


def cleanup_inactive_users() -> None:
    """Очистка неактивных пользователей"""
    now = datetime.now()
    to_delete = [
        uid for uid, ctx in user_contexts.items()
        if now - ctx.get("last_activity", now) > timedelta(hours=INACTIVITY_LIMIT_HOURS)
    ]
    for uid in to_delete:
        del user_contexts[uid]


def save_question_for_answer(user_id: int, answer_index: int, question: str) -> None:
    """Сохранение вопроса для конкретного ответа"""
    ctx = get_user_context(user_id)
    ctx["question_index_map"][answer_index] = question


def get_question_for_answer(user_id: int, answer_index: int) -> str:
    """Получение вопроса для конкретного ответа"""
    ctx = get_user_context(user_id)
    return ctx.get("question_index_map", {}).get(answer_index, "???")


def get_contextual_question(user_id: int, current_question: str) -> str:
    """Добавляет контекст для уточняющих вопросов"""
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


# ============================================================
# 📱 ОБРАБОТЧИКИ КОМАНД
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /start"""
    user_id = update.effective_user.id
    
    cleanup_inactive_users()
    
    is_returning = user_id in user_contexts
    is_admin = (user_id == ADMIN_USER_ID)
    
    get_user_context(user_id)
    update_user_activity(user_id)
    
    if is_returning:
        text = AppleStyleMessages.WELCOME_RETURNING
    else:
        text = AppleStyleMessages.WELCOME
    
    await update.message.reply_text(
        text, 
        reply_markup=AppleKeyboards.main_menu(is_returning, is_admin),
        parse_mode="HTML"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка команды /help"""
    await update.message.reply_text(
        AppleStyleMessages.HELP, 
        parse_mode="HTML"
    )


async def roadmaps_command(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                           edit_mode: bool = False) -> None:
    """Обработка команды /roadmaps"""
    text = "🗺 <b>Дорожные карты обучения</b>\n\nВыберите направление:"
    
    if edit_mode and update.callback_query:
        await update.callback_query.edit_message_text(
            text, 
            reply_markup=AppleKeyboards.roadmaps_menu(), 
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            text, 
            reply_markup=AppleKeyboards.roadmaps_menu(), 
            parse_mode="HTML"
        )


# ============================================================
# 🎯 ОБРАБОТЧИК CALLBACK-КНОПОК
# ============================================================

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Центральный обработчик всех callback-кнопок"""
    query = update.callback_query
    data = query.data
    
    await query.answer()
    
    user_id = update.effective_user.id
    is_admin = (user_id == ADMIN_USER_ID)
    update_user_activity(user_id)
    
    # --- НАВИГАЦИЯ ПО МЕНЮ ---
    
    if data == "menu_main":
        await query.edit_message_text(
            AppleStyleMessages.WELCOME_RETURNING,
            reply_markup=AppleKeyboards.main_menu(is_returning=True, is_admin=is_admin),
            parse_mode="HTML"
        )
        return
    
    if data == "menu_consult":
        text = "🗓 <b>Запись на консультацию</b>\n\nВыберите удобный способ:"
        await query.edit_message_text(
            text,
            reply_markup=AppleKeyboards.consult_menu(),
            parse_mode="HTML"
        )
        return
    
    if data == "menu_roadmaps":
        await roadmaps_command(update, context, edit_mode=True)
        return
    
    # --- АДМИН-ПАНЕЛЬ (НОВАЯ ЛОГИКА) ---
    
    if data == "admin_panel" and is_admin:
        text = "⚙️ <b>Панель управления</b>\n\nВыберите раздел для просмотра данных:"
        await query.edit_message_text(
            text,
            reply_markup=AppleKeyboards.admin_panel(),
            parse_mode="HTML"
        )
        return
    
    if data.startswith("admin_page_") and is_admin:
        parts = data.split("_")
        await admin_show_list(update, context, parts[2], int(parts[3]))
        return
    
    if data.startswith("admin_clear_") and is_admin:
        await admin_clear_confirm(update, context, data.replace("admin_clear_", ""))
        return
    
    if data.startswith("admin_do_clear_") and is_admin:
        await admin_do_clear(update, context, data.replace("admin_do_clear_", ""))
        return
    
    # --- СТАНДАРТНЫЕ ВОПРОСЫ МЕНЮ ---
    
    if data in ["menu_cost", "menu_method", "menu_about"]:
        q_map = {
            "menu_cost": "стоимость", 
            "menu_method": "метод выстраданного познания", 
            "menu_about": "кто такой алексей"
        }
        
        if not kb_index:
            await query.edit_message_text(
                "⚠️ База знаний недоступна",
                reply_markup=AppleKeyboards.back_button()
            )
            return
        
        answer, score, candidates = search_knowledge_base(q_map[data], kb_index)
        
        if not answer:
            await query.edit_message_text(
                AppleStyleMessages.NOT_FOUND,
                reply_markup=AppleKeyboards.back_button(),
                parse_mode="HTML"
            )
            return
        
        clean_text = answer.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        
        ans_idx = 0
        if candidates:
            ans_idx = candidates[0]['index']
        else:
            for i, item in enumerate(kb_index.items):
                if item['context'] == answer:
                    ans_idx = i
                    break
        
        save_question_for_answer(user_id, ans_idx, q_map[data])
        
        if "[add_button]" in answer:
            url_buttons.append([
                InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")
            ])
        
        url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))
        
        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            disable_web_page_preview=True,
            parse_mode="HTML"
        )
        return
    
    # --- УТОЧНЕНИЕ ВОПРОСА ---
    
    if data.startswith("clarify_"):
        if data == "clarify_none":
            await query.edit_message_text(
                "Хорошо, попробуйте сформулировать иначе.",
                reply_markup=AppleKeyboards.back_button()
            )
            return
        
        idx = int(data.split("_")[1])
        
        if not kb_index or not kb_index.is_valid_index(idx):
            await query.answer("Ответ не найден", show_alert=True)
            return
        
        context_data = kb_index.items[idx]["context"]
        clean_text = context_data.replace("[add_button]", "").strip()
        display_text, url_buttons = extract_links_and_buttons(clean_text)
        
        if "[add_button]" in context_data:
            url_buttons.append([
                InlineKeyboardButton("📝 Записаться", callback_data="consultation")
            ])
        
        save_question_for_answer(user_id, idx, "Уточняющий вопрос")
        
        url_buttons.extend(AppleKeyboards.feedback_buttons(idx))
        
        await query.edit_message_text(
            display_text,
            reply_markup=InlineKeyboardMarkup(url_buttons),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        return
    
    # --- КОНСУЛЬТАЦИЯ ---
    
    if data == "consultation":
        await consultation_callback(update, context)
        return
    
    # --- ОБРАТНАЯ СВЯЗЬ ---
    
    if data.startswith("like_") or data.startswith("dislike_"):
        await feedback_callback(update, context)
        return
    
    if data == "ignore":
        return


# ============================================================
# 📝 КОНСУЛЬТАЦИЯ
# ============================================================

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка заявки на консультацию"""
    query = update.callback_query
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    consultations = load_json(CONSULTATIONS_FILE)
    recent_consultations = [
        c for c in consultations
        if c.get("user_id") == user.id and
        datetime.now() - datetime.strptime(c.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S") < timedelta(hours=24)
    ]
    
    if recent_consultations:
        await query.edit_message_text(
            "✅ <b>Вы уже записаны</b>\n\nВаша заявка обрабатывается. Ожидайте связи!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📅 Календарь", url=CALENDAR_URL)]
            ]),
            parse_mode="HTML"
        )
        return
    
    consultations.append({
        "user_id": user.id,
        "username": user.username or "Нет",
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "timestamp": timestamp
    })
    save_json(CONSULTATIONS_FILE, consultations)
    
    try:
        await context.bot.send_message(
            ADMIN_USER_ID,
            f"🔔 <b>Новая заявка!</b>\n\n"
            f"👤 {user.first_name or 'Без имени'}\n"
            f"📱 @{user.username or 'нет username'}\n"
            f"🆔 {user.id}",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Failed to notify admin: {e}")
    
    keyboard = [[InlineKeyboardButton("📅 Выбрать время в календаре", url=CALENDAR_URL)]]
    await query.edit_message_text(
        AppleStyleMessages.CONSULTATION_SUCCESS,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


# ============================================================
# 💚 ОБРАТНАЯ СВЯЗЬ
# ============================================================

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка лайков/дизлайков"""
    query = update.callback_query
    data = query.data
    user = query.from_user
    
    await query.answer()
    
    fb_type = "like" if data.startswith("like_") else "dislike"
    
    try:
        idx = int(data.split("_")[1])
    except (IndexError, ValueError) as e:
        logger.error(f"Invalid callback data format: {data}, error: {e}")
        await query.answer("Ошибка данных", show_alert=True)
        return
    
    if not kb_index:
        logger.error("kb_index is None")
        await query.answer("База недоступна", show_alert=True)
        return
    
    if not kb_index.is_valid_index(idx):
        logger.error(f"Index {idx} out of bounds")
        await query.answer("Ответ не найден", show_alert=True)
        return
    
    answer = kb_index.items[idx]["context"]
    question = get_question_for_answer(user.id, idx)
    
    if question == "???":
        ctx = get_user_context(user.id)
        history = ctx.get("history", [])
        if history:
            question = list(history)[-1]
    
    feedback_list = load_json(FEEDBACK_FILE)
    feedback_list.append({
        "type": fb_type,
        "question": question,
        "answer": answer[:200] + "..." if len(answer) > 200 else answer,
        "user_id": user.id,
        "username": user.username,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_json(FEEDBACK_FILE, feedback_list)
    
    if fb_type == "like":
        new_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💚 Спасибо за оценку!", callback_data="ignore")]
        ])
        await query.edit_message_reply_markup(new_keyboard)
    else:
        new_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📝 Жалоба отправлена", callback_data="ignore")]
        ])
        await query.edit_message_reply_markup(new_keyboard)
        await query.message.reply_text(AppleStyleMessages.FEEDBACK_DISLIKE, parse_mode="HTML")
        
        try:
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"👎 <b>Дизлайк</b>\n\n"
                f"❓ <b>Вопрос:</b> {question}\n"
                f"💬 <b>Ответ:</b> {answer[:100]}...\n"
                f"👤 @{user.username or user.id}",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Failed to notify admin: {e}")


# ============================================================
# 💬 ГЛАВНЫЙ ОБРАБОТЧИК СООБЩЕНИЙ
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка текстовых сообщений"""
    
    if not update.message or not update.message.text:
        return
    
    user_id = update.effective_user.id
    user_question = update.message.text.strip()
    
    # ✅ УБРАНО: Текстовые команды админа. Теперь только кнопки.
    
    cleanup_inactive_users()
    
    get_user_context(user_id)
    update_user_activity(user_id)
    user_contexts[user_id]["history"].append(user_question)
    
    search_query = get_contextual_question(user_id, user_question)
    answer, score, candidates = search_knowledge_base(search_query, kb_index)
    final_answer = None
    
    if score > 3.5 and answer:
        final_answer = answer
    elif score > 1.5 and candidates:
        keyboard = [
            [InlineKeyboardButton(f"💬 {c['topic']}", callback_data=f"clarify_{c['index']}")]
            for c in candidates
        ]
        keyboard.append([InlineKeyboardButton("❌ Не то", callback_data="clarify_none")])
        
        await update.message.reply_text(
            AppleStyleMessages.CLARIFY_PROMPT,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )
        return
    elif FUZZY_ENABLED:
        suggestion = get_fuzzy_suggestion(user_question, kb_index)
        if suggestion:
            answer, score, candidates = search_knowledge_base(suggestion, kb_index)
            if score > 1.5:
                final_answer = answer
            if score < 3.5 and candidates:
                keyboard = [
                    [InlineKeyboardButton(f"💡 {suggestion}?", callback_data=f"clarify_{candidates[0]['index']}")]
                ]
                await update.message.reply_text(
                    AppleStyleMessages.FUZZY_SUGGESTION,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode="HTML"
                )
                return
    
    if not final_answer:
        unk = load_json(UNKNOWN_FILE)
        unk.append({
            "question": user_question,
            "user_id": user_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_json(UNKNOWN_FILE, unk)
        
        # Проверяем, админ ли пишет, чтобы показать ему нужную клавиатуру
        is_admin = (user_id == ADMIN_USER_ID)
        await update.message.reply_text(
            AppleStyleMessages.NOT_FOUND,
            reply_markup=AppleKeyboards.main_menu(is_returning=True, is_admin=is_admin),
            parse_mode="HTML"
        )
        return
    
    clean_answer = final_answer.replace("[add_button]", "").strip()
    user_contexts[user_id]["last_answer"] = clean_answer
    
    display_text, url_buttons = extract_links_and_buttons(clean_answer)
    
    ans_idx = 0
    if candidates and candidates[0]['context'] == final_answer:
        ans_idx = candidates[0]['index']
    else:
        for i, item in enumerate(kb_index.items):
            if item['context'] == final_answer:
                ans_idx = i
                break
    
    save_question_for_answer(user_id, ans_idx, user_question)
    
    if "[add_button]" in final_answer:
        url_buttons.append([
            InlineKeyboardButton("📝 Записаться на консультацию", callback_data="consultation")
        ])
    
    url_buttons.extend(AppleKeyboards.feedback_buttons(ans_idx))
    
    await update.message.reply_text(
        display_text,
        reply_markup=InlineKeyboardMarkup(url_buttons),
        disable_web_page_preview=True,
        parse_mode="HTML"
    )


# ============================================================
# 👨‍💼 АДМИН-ПАНЕЛЬ (ОТОБРАЖЕНИЕ СПИСКОВ)
# ============================================================

async def admin_show_list(update: Update, context: ContextTypes.DEFAULT_TYPE, 
                          data_type: str, page: int = 0):
    """Отображение списка данных для админа"""
    query = update.callback_query
    if query:
        await query.answer()
    
    items = []
    title = ""
    empty_msg = ""
    clear_callback = ""
    
    if data_type == "consult":
        items = load_json(CONSULTATIONS_FILE)
        title = "📋 Заявки на консультацию"
        empty_msg = "Заявок пока нет."
        clear_callback = "admin_clear_consult"
    elif data_type == "like":
        all_fb = load_json(FEEDBACK_FILE)
        items = [x for x in all_fb if x.get("type") == "like"]
        title = "💚 Лайки"
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
        empty_msg = "Бот знает ответы на все вопросы."
        clear_callback = "admin_clear_unknown"
    
    total_items = len(items)
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
    
    if page < 0:
        page = 0
    if page >= total_pages:
        page = total_pages - 1
    
    text = f"<b>{title}</b>\nВсего: {total_items}\n\n"
    
    if not items:
        text += f"<i>{empty_msg}</i>"
    else:
        start_idx = page * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        current_items = items[start_idx:end_idx]
        
        for i, item in enumerate(current_items, start=start_idx + 1):
            if data_type == "consult":
                text += f"{i}. {item.get('first_name', '')} @{item.get('username', '')}\n   ⏰ {item.get('timestamp', '')}\n\n"
            elif data_type == "unknown":
                text += f"{i}. {item.get('question', '???')}\n\n"
            else:
                q = item.get('question', '???')
                text += f"{i}. {q[:50]}{'...' if len(q) > 50 else ''}\n\n"
    
    keyboard = []
    
    # Навигация
    if total_pages > 1:
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("◀️", callback_data=f"admin_page_{data_type}_{page-1}"))
        nav_row.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="ignore"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("▶️", callback_data=f"admin_page_{data_type}_{page+1}"))
        keyboard.append(nav_row)
    
    if items:
        keyboard.append([InlineKeyboardButton("🗑 Очистить список", callback_data=clear_callback)])
    
    keyboard.append([InlineKeyboardButton("◀️ Назад в панель", callback_data="admin_panel")])
    
    markup = InlineKeyboardMarkup(keyboard)
    
    if query:
        try:
            await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except Exception:
            pass
    else:
        await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")


async def admin_clear_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    """Подтверждение очистки"""
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton("✅ Да, очистить", callback_data=f"admin_do_clear_{data_type}")],
        [InlineKeyboardButton("❌ Отмена", callback_data=f"admin_page_{data_type}_0")]
    ]
    
    await query.edit_message_text(
        "⚠️ <b>Подтвердите очистку</b>\n\nЭто действие нельзя отменить.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML"
    )


async def admin_do_clear(update: Update, context: ContextTypes.DEFAULT_TYPE, data_type: str):
    """Выполнение очистки"""
    query = update.callback_query
    await query.answer()
    
    if data_type == "consult":
        save_json(CONSULTATIONS_FILE, [])
    elif data_type in ["like", "dislike"]:
        fb = load_json(FEEDBACK_FILE)
        save_json(FEEDBACK_FILE, [x for x in fb if x.get("type") != data_type])
    elif data_type == "unknown":
        save_json(UNKNOWN_FILE, [])
    
    await query.edit_message_text("✅ <b>Очищено успешно</b>", parse_mode="HTML")


# ============================================================
# ⚠️ ОБРАБОТЧИК ОШИБОК
# ============================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок"""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Что-то пошло не так.\n\nПопробуйте позже или напишите /start",
                parse_mode="HTML"
            )
        except Exception:
            pass
    
    if ADMIN_USER_ID:
        try:
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            
            await context.bot.send_message(
                ADMIN_USER_ID,
                f"❌ <b>ERROR:</b>\n<pre>{tb_string[:4000]}</pre>",
                parse_mode="HTML"
            )
        except Exception:
            pass


# ============================================================
# 🚀 ЗАПУСК
# ============================================================

def main() -> None:
    """Точка входа"""
    global kb_index
    
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("❌ Токен не найден в переменных окружения BOT_TOKEN")
    
    try:
        kb = load_knowledge_base('main.json')
        kb_index = preprocess_knowledge_base(kb)
        print(f"✅ База знаний загружена: {len(kb_index.items)} записей")
    except Exception as e:
        print(f"❌ Ошибка загрузки базы знаний: {str(e)}")
        return
    
    application = Application.builder().token(token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("roadmaps", roadmaps_command))
    application.add_handler(CallbackQueryHandler(menu_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.add_error_handler(error_handler)
    
    print("🚀 Бот запущен")
    application.run_polling()


if __name__ == "__main__":
    main()