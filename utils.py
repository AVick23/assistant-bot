import json
import re
import os
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    RUSSIAN_STOPWORDS, SYNONYMS, logger,
    MIN_FULLTEXT_SCORE, MAX_HISTORY_LENGTH, INACTIVITY_LIMIT_HOURS
)

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()

# Глобальные переменные для хранения индекса и контекстов пользователей
kb_index = None  # будет инициализирован в main.py
user_contexts: Dict[int, dict] = {}


# ====================== Работа с JSON ======================

def load_json(file_path: str) -> list:
    """Безопасная загрузка JSON-файла"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def save_json(file_path: str, data: list) -> None:
    """Безопасное сохранение JSON-файла"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error(f"Error saving {file_path}: {e}")


# ====================== NLP функции ======================

def preprocess_question(question: str) -> str:
    """Удаляет вводные фразы (а если, скажи и т.п.)"""
    patterns = [
        r'^а если\s+', r'^что если\s+', r'^что будет если\s+',
        r'^можно ли\s+', r'^а что если\s+', r'^если я\s+',
        r'^а\s+', r'^ну\s+', r'^скажи\s+', r'^расскажи\s+', r'^объясни\s+'
    ]
    cleaned = question.lower()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    return cleaned.strip()

def preprocess_text(text: str) -> str:
    """Очистка текста от ссылок, email и знаков препинания"""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    """Лемматизация одного слова с кэшированием"""
    if not hasattr(lemmatize_word, 'cache'):
        lemmatize_word.cache = {}
    if word in lemmatize_word.cache:
        return lemmatize_word.cache[word]
    parsed = morph.parse(word)[0]
    lemma = parsed.normal_form
    lemmatize_word.cache[word] = lemma
    return lemma

def lemmatize_sentence(text: str) -> str:
    """Лемматизация предложения с удалением стоп-слов и коротких слов"""
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(word) for word in words
              if word not in RUSSIAN_STOPWORDS and len(word) > 2]
    return " ".join(lemmas)

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Расширение множества ключевых слов синонимами"""
    expanded = set(keywords)
    for word in keywords:
        for base, synonyms in SYNONYMS.items():
            if word == base or any(word == syn for syn in synonyms):
                expanded.update([base] + synonyms)
    return expanded

def extract_keywords(text: str, use_synonyms: bool = True) -> Set[str]:
    """Извлечение ключевых слов из текста"""
    cleaned = preprocess_text(text)
    words = cleaned.split()
    keywords = {lemmatize_word(word) for word in words
                if len(word) > 2 and word not in RUSSIAN_STOPWORDS}
    if use_synonyms and keywords:
        keywords = expand_with_synonyms(keywords)
    return keywords

def calculate_keyword_match_score(user_keywords: Set[str], item_keywords: Set[str],
                                  user_question: str, original_keywords: List[str]) -> float:
    """Оценка совпадения по ключевым словам"""
    common = user_keywords.intersection(item_keywords)
    base_score = len(common) * 2
    question_lower = preprocess_text(user_question)
    phrase_bonus = 0
    for kw in original_keywords:
        kw_lower = preprocess_text(kw)
        if kw_lower in question_lower:
            phrase_bonus += len(kw_lower.split()) * 3
    return base_score + phrase_bonus


# ====================== Класс индекса базы знаний ======================

class KBIndex:
    def __init__(self):
        self.items = []                # список словарей с ключами "context", "keywords", "original_keywords"
        self.contexts = []              # только тексты ответов
        self.tfidf_vectorizer = None
        self.tfidf_labeled_matrix = None
        self.raw_tfidf_vectorizer = None
        self.tfidf_raw_matrix = None
        self.all_keywords_list = []     # для нечёткого поиска

    def build_tfidf_index(self, contexts: List[str]):
        """Строит два TF-IDF индекса: по лемматизированным текстам и по сырым"""
        # Индекс по лемматизированным текстам (для лучшего semantic matching)
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 3),
            max_features=3000
        )
        lemmatized = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized)

        # Индекс по сырым текстам (для учёта точных фраз)
        self.raw_tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 2),
            max_features=2000
        )
        self.tfidf_raw_matrix = self.raw_tfidf_vectorizer.fit_transform(contexts)

        # Собираем все ключевые слова для нечёткого поиска
        all_kw = set()
        for item in self.items:
            all_kw.update(item["original_keywords"])
        self.all_keywords_list = list(all_kw)

    def keyword_search(self, user_question: str, top_k: int = 3) -> List[dict]:
        """Поиск по ключевым словам (точное совпадение с бонусами)"""
        user_keywords = extract_keywords(user_question)
        if not user_keywords:
            return []

        scored = []
        for idx, item in enumerate(self.items):
            score = calculate_keyword_match_score(
                user_keywords, item["keywords"], user_question, item["original_keywords"]
            )
            if score > 0:
                scored.append({
                    "context": item["context"],
                    "score": score,
                    "index": idx
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def fulltext_search(self, query: str, top_k: int = 3) -> List[dict]:
        """TF-IDF поиск (комбинация лемматизированного и сырого)"""
        if self.tfidf_vectorizer is None or self.tfidf_labeled_matrix is None:
            return []

        try:
            # Лемматизированный запрос
            query_lemma = lemmatize_sentence(query)
            query_vec = self.tfidf_vectorizer.transform([query_lemma])
            labeled_sim = cosine_similarity(query_vec, self.tfidf_labeled_matrix)[0]

            # Сырой запрос
            raw_vec = self.raw_tfidf_vectorizer.transform([query])
            raw_sim = cosine_similarity(raw_vec, self.tfidf_raw_matrix)[0]

            # Взвешенная сумма
            combined = 0.7 * labeled_sim + 0.3 * raw_sim
            top_indices = np.argsort(combined)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = combined[idx]
                if score > MIN_FULLTEXT_SCORE:
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
    """Создаёт экземпляр KBIndex из сырой базы знаний (main.json)"""
    kb_index = KBIndex()
    processed_items = []
    contexts = [item["context"] for item in knowledge_base]

    for item in knowledge_base:
        # Обработка ключевых слов: лемматизация, удаление стоп-слов
        processed_keywords = set()
        for keyword in item["keywords"]:
            for word in re.split(r'\s+', preprocess_text(keyword)):
                if len(word) > 2 and word not in RUSSIAN_STOPWORDS:
                    processed_keywords.add(lemmatize_word(word))
        processed_items.append({
            "context": item["context"],
            "keywords": processed_keywords,
            "original_keywords": item["keywords"]
        })

    kb_index.items = processed_items
    kb_index.contexts = contexts
    kb_index.build_tfidf_index(contexts)
    return kb_index


def search_knowledge_base(user_question: str, kb_index: KBIndex) -> Tuple[Optional[str], float, List[dict]]:
    """
    Комбинированный поиск: объединяет результаты keyword_search и fulltext_search
    с весами и возвращает лучший ответ (если уверенность высока) или список кандидатов.
    """
    cleaned = preprocess_question(user_question)

    kw_results = kb_index.keyword_search(cleaned, top_k=5)
    ft_results = kb_index.fulltext_search(cleaned, top_k=5)

    # Если по очищенному запросу ничего не нашлось, пробуем исходный
    if not kw_results and not ft_results:
        kw_results = kb_index.keyword_search(user_question, top_k=5)
        ft_results = kb_index.fulltext_search(user_question, top_k=5)

    # Собираем общую оценку по индексам
    combined = {}
    for res in kw_results:
        combined.setdefault(res["index"], 0)
        combined[res["index"]] += res["score"] * 0.6

    for res in ft_results:
        combined.setdefault(res["index"], 0)
        combined[res["index"]] += res["score"] * 50 * 0.4  # Масштабируем TF-IDF до сопоставимых величин

    if not combined:
        return None, 0.0, []

    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    candidates = []
    for idx, score in sorted_items[:3]:
        topic = kb_index.items[idx]["original_keywords"][0] if kb_index.items[idx]["original_keywords"] else "Тема"
        candidates.append({
            "index": idx,
            "score": score,
            "topic": topic,
            "context": kb_index.items[idx]["context"]
        })

    best_idx, best_score = sorted_items[0]
    return kb_index.items[best_idx]["context"], best_score, candidates


# ====================== Управление контекстом пользователя ======================

def get_user_context(user_id: int) -> dict:
    """Возвращает словарь контекста для пользователя, создаёт при необходимости"""
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=MAX_HISTORY_LENGTH),
            "last_activity": datetime.now(),
            "question_index_map": {},   # связь индекса ответа с заданным вопросом
            "favorites": set()           # новые: избранные ответы (индексы)
        }
    return user_contexts[user_id]

def update_user_activity(user_id: int) -> None:
    get_user_context(user_id)["last_activity"] = datetime.now()

def cleanup_inactive_users() -> None:
    now = datetime.now()
    to_delete = [
        uid for uid, ctx in user_contexts.items()
        if now - ctx.get("last_activity", now) > timedelta(hours=INACTIVITY_LIMIT_HOURS)
    ]
    for uid in to_delete:
        del user_contexts[uid]

def save_question_for_answer(user_id: int, answer_index: int, question: str) -> None:
    ctx = get_user_context(user_id)
    ctx["question_index_map"][answer_index] = question

def get_question_for_answer(user_id: int, answer_index: int) -> str:
    ctx = get_user_context(user_id)
    return ctx.get("question_index_map", {}).get(answer_index, "???")

def add_favorite(user_id: int, answer_index: int) -> None:
    ctx = get_user_context(user_id)
    ctx["favorites"].add(answer_index)

def remove_favorite(user_id: int, answer_index: int) -> None:
    ctx = get_user_context(user_id)
    ctx["favorites"].discard(answer_index)

def get_favorites(user_id: int) -> List[int]:
    return list(get_user_context(user_id)["favorites"])

def get_contextual_question(user_id: int, current_question: str) -> str:
    """Добавляет последнее сообщение из истории, если текущий вопрос короткий или уточняющий"""
    ctx = get_user_context(user_id)
    history = ctx.get("history", [])
    if not history:
        return current_question

    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'а скидки', 'а рассрочка', 'а документ']
    q_lower = current_question.lower()
    if len(q_lower) < 20 or any(marker in q_lower for marker in context_markers):
        last_msg = list(history)[-1]
        return f"{last_msg} {current_question}"
    return current_question


# ====================== Извлечение ссылок и кнопок ======================

def extract_links_and_buttons(text: str) -> Tuple[str, List[List[Any]]]:
    """Ищет ссылки в тексте и возвращает чистый текст и список рядов InlineKeyboardButton"""
    from telegram import InlineKeyboardButton  # импорт здесь, чтобы избежать циклического импорта

    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)

    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "").strip('.,;:!?()"\'[]{}')
            if not clean_url:
                continue

            # Красивые подписи
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

        # Убираем ссылки из текста
        clean_text = re.sub(url_pattern, '', text).strip()
        clean_text = re.sub(r'\s+\.', '.', clean_text)
        clean_text = re.sub(r'\(\s*\)', '', clean_text).strip()
        return clean_text, buttons

    return text, []