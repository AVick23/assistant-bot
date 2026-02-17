import json
import re
import os
import logging
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
from rank_bm25 import BM25Okapi
import numpy as np
from config import (
    FILES, RUSSIAN_STOPWORDS, SYNONYMS, morph,
    logger, SETTINGS, URLS
)

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
_kb_index = None
user_contexts: Dict[int, dict] = {}

# ============================================================
# ГЕТТЕР ДЛЯ KB_INDEX
# ============================================================
def get_kb_index() -> 'KBIndex':
    return _kb_index

# ============================================================
# РАБОТА С JSON
# ============================================================
def load_json(file_path: str) -> list:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def save_json(file_path: str, data: list) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")

# ============================================================
# NLP УТИЛИТЫ
# ============================================================
def preprocess_text(text: str) -> str:
    """Очистка текста"""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    """Лемматизация с кэшированием"""
    if not hasattr(lemmatize_word, 'cache'):
        lemmatize_word.cache = {}
    if word in lemmatize_word.cache:
        return lemmatize_word.cache[word]
    try:
        parsed = morph.parse(word)[0]
        lemma = parsed.normal_form
        lemmatize_word.cache[word] = lemma
        return lemma
    except:
        return word

def lemmatize_sentence(text: str) -> str:
    """Лемматизация всего предложения"""
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(w) for w in words if w not in RUSSIAN_STOPWORDS and len(w) > 2]
    return " ".join(lemmas)

def expand_query_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Расширение запроса пользователя синонимами"""
    expanded = set(keywords)
    for word in keywords:
        for base, syns in SYNONYMS.items():
            if word == base or word in syns:
                expanded.add(base)
                expanded.update(syns)
    return expanded

# ============================================================
# КЛАСС ИНДЕКСА (HYBRID SEARCH: Keywords + Context)
# ============================================================
class KBIndex:
    def __init__(self, items: list):
        self.items = items
        self.contexts = [item['context'] for item in items] if items else []
        
        # ✅ Подготовка данных для BM25
        # Используем context И keywords из JSON
        searchable_texts = []
        for item in items:
            text = item['context']
            # Добавляем keywords к тексту для индексации, чтобы BM25 их учитывал
            if item.get('keywords'):
                text += " " + " ".join(item['keywords'])
            searchable_texts.append(text)
        
        self.tokenized_contexts = [lemmatize_sentence(t).split() for t in searchable_texts] if searchable_texts else []
        self.bm25 = BM25Okapi(self.tokenized_contexts) if self.tokenized_contexts else None
        
        # Сохраняем списки ключевых слов для точного матчинга
        self.item_keywords = [set(item.get('keywords', [])) for item in items]
    
    def search(self, query: str, top_k: int = 5, user_context: Optional[dict] = None) -> List[dict]:
        """
        ✅ Гибридный поиск: BM25 + Keywords + Контекст беседы
        """
        if not self.items or not self.bm25:
            return []
        
        query_lemmas = lemmatize_sentence(query).split()
        query_lower = preprocess_text(query)
        
        # --- Шаг 1: Базовые оценки BM25 ---
        bm25_scores = self.bm25.get_scores(query_lemmas)
        
        # --- Шаг 2: Бонусы за Keywords (Точное совпадение) ---
        final_scores = bm25_scores.copy()
        
        for idx in range(len(self.items)):
            score_boost = 0.0
            
            # Проверяем ключевые слова записи
            for kw in self.item_keywords[idx]:
                # Если ключевое слово (фраза) найдено в запросе пользователя
                if len(kw.split()) > 1 and kw.lower() in query_lower:
                    score_boost += 5.0  # Большой бонус за фразу
                elif kw.lower() in query_lower:
                    score_boost += 2.0  # Бонус за слово
            
            # ✅ Бонус за контекст беседы
            if user_context:
                history = user_context.get('history', [])
                history_list = list(history)
                for hist_msg in history_list[-5:]:
                    hist_lemmas = set(lemmatize_sentence(hist_msg).split())
                    query_lemmas_set = set(query_lemmas)
                    overlap = len(hist_lemmas & query_lemmas_set)
                    if overlap > 0:
                        score_boost += overlap * 0.5
            
            final_scores[idx] += score_boost
        
        # --- Шаг 3: Сортировка и отбор ---
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = final_scores[idx]
            if score > 0.5:
                results.append({
                    "index": int(idx),
                    "score": float(score),
                    "context": self.contexts[idx],
                    "topic": self.items[idx].get('keywords', ['Тема'])[0] if self.items[idx].get('keywords') else 'Тема'
                })
        
        return results
    
    def is_valid_index(self, idx: int) -> bool:
        return 0 <= idx < len(self.items)

# ============================================================
# ИНИЦИАЛИЗАЦИЯ БАЗЫ ЗНАНИЙ
# ============================================================
def initialize_kb() -> KBIndex:
    """
    Инициализация базы знаний.
    ✅ Просто загружает JSON и создает индекс. Ничего не генерирует.
    """
    global _kb_index
    
    kb_data = load_json(FILES['kb'])
    
    if not kb_data:
        logger.error("❌ База знаний пуста или файл не найден!")
        _kb_index = KBIndex([])
        return _kb_index
    
    # ✅ Проверка наличия keywords (просто для информации)
    count_with_kw = sum(1 for item in kb_data if item.get('keywords'))
    if count_with_kw < len(kb_data):
        logger.warning(f"⚠️ Внимание: {len(kb_data) - count_with_kw} записей не имеют ключевых слов!")
    
    # Создание индекса
    _kb_index = KBIndex(kb_data)
    
    logger.info(f"✅ База знаний загружена: {len(_kb_index.items)} записей")
    return _kb_index

# ============================================================
# ПОИСК И КОНТЕКСТ
# ============================================================
def get_user_context(user_id: int) -> dict:
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=SETTINGS['max_history']),
            "last_activity": datetime.now(),
            "question_index_map": {}
        }
    return user_contexts[user_id]

def update_user_activity(user_id: int):
    ctx = get_user_context(user_id)
    ctx["last_activity"] = datetime.now()

def save_question_for_answer(user_id: int, ans_idx: int, question: str):
    ctx = get_user_context(user_id)
    ctx["question_index_map"][ans_idx] = question

def get_question_for_answer(user_id: int, ans_idx: int) -> str:
    ctx = get_user_context(user_id)
    return ctx.get("question_index_map", {}).get(ans_idx, "???")

def save_message_to_history(user_id: int, message: str, is_user: bool = True):
    ctx = get_user_context(user_id)
    prefix = "User: " if is_user else "Bot: "
    ctx["history"].append(f"{prefix}{message}")

def get_contextual_question(user_id: int, current_question: str) -> str:
    ctx = get_user_context(user_id)
    history = ctx.get("history", [])
    
    if not history:
        return current_question
    
    history_list = list(history)
    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'а скидки', 'а рассрочка', 'а документ', 
                       'и', 'тоже', 'также', 'еще', 'ещё', 'продолжи', 'далее']
    q_lower = current_question.lower()
    
    if len(q_lower) < 20 or any(marker in q_lower for marker in context_markers):
        recent_history = history_list[-3:] if len(history_list) >= 3 else history_list
        history_context = " ".join(recent_history)
        return f"{history_context} {current_question}"
    
    return current_question

def cleanup_inactive_users():
    now = datetime.now()
    to_delete = [
        uid for uid, ctx in user_contexts.items()
        if now - ctx.get("last_activity", now) > timedelta(hours=SETTINGS['inactivity_hours'])
    ]
    for uid in to_delete:
        del user_contexts[uid]

# ============================================================
# ИЗВЛЕЧЕНИЕ ССЫЛОК ДЛЯ КНОПОК
# ============================================================
def extract_links_and_buttons(text: str) -> Tuple[str, List[List[dict]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)
    
    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "").strip('.,;:!?() "\'[]{}')
            if not clean_url:
                continue
            
            label = "🔗 Ссылка"
            if "roadmap" in clean_url.lower():
                label = "🗺 Дорожная карта"
            elif "Business-card" in clean_url:
                label = "👤 О преподавателе"
            elif "calendar" in clean_url.lower():
                label = "📅 Выбрать время"
            
            buttons.append([{"text": label, "url": clean_url}])
        
        clean_text = re.sub(url_pattern, '', text).strip()
        return clean_text, buttons
    
    return text, []