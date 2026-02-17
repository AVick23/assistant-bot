import json
import re
import math
import os
import logging
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque, Counter
from rank_bm25 import BM25Okapi
import numpy as np
from config import (
    FILES, RUSSIAN_STOPWORDS, SYNONYMS, morph,
    logger, SETTINGS, URLS
)

# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
_kb_index = None  # ✅ Приватная переменная
user_contexts: Dict[int, dict] = {}

# ============================================================
# ГЕТТЕР ДЛЯ KB_INDEX (решает проблему с None)
# ============================================================
def get_kb_index() -> 'KBIndex':
    """Возвращает текущий индекс базы знаний"""
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

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Расширение синонимами"""
    expanded = set(keywords)
    for word in keywords:
        for base, syns in SYNONYMS.items():
            if word == base or word in syns:
                expanded.add(base)
                expanded.update(syns)
    return expanded

# ============================================================
# АВТО-ГЕНЕРАЦИЯ KEYWORDS
# ============================================================
def auto_generate_keywords(context: str, max_kw: int = None) -> List[str]:
    """
    Автоматически извлекает ключевые слова из текста.
    """
    if max_kw is None:
        max_kw = SETTINGS.get('max_keywords', 30)
    
    keywords = set()
    
    # 1. Берем первые 3 предложения (суть)
    sentences = re.split(r'[.!?]', context)[:3]
    important_text = ' '.join(sentences)
    words = preprocess_text(important_text).split()
    
    for word in words:
        if len(word) > 2 and word not in RUSSIAN_STOPWORDS:
            try:
                parsed = morph.parse(word)[0]
                if any(tag in parsed.tag for tag in ['NOUN', 'ADJF', 'INFN', 'VERB']):
                    keywords.add(parsed.normal_form)
            except:
                keywords.add(word)
    
    # 2. Добавляем слова из ВСЕГО контекста
    full_words = preprocess_text(context).split()
    for word in full_words:
        if len(word) > 2 and word not in RUSSIAN_STOPWORDS:
            try:
                parsed = morph.parse(word)[0]
                if any(tag in parsed.tag for tag in ['NOUN', 'ADJF', 'INFN', 'VERB']):
                    keywords.add(parsed.normal_form)
            except:
                keywords.add(word)
    
    # 3. Добавляем фразы (2-3 слова)
    word_list = preprocess_text(context).split()
    for i in range(len(word_list) - 1):
        phrase_2 = f"{word_list[i]} {word_list[i+1]}"
        if len(phrase_2) > 5:
            keywords.add(phrase_2)
    
    for i in range(len(word_list) - 2):
        phrase_3 = f"{word_list[i]} {word_list[i+1]} {word_list[i+2]}"
        if len(phrase_3) > 8:
            keywords.add(phrase_3)
    
    # 4. Расширяем синонимами
    keywords = expand_with_synonyms(keywords)
    
    # 5. Специальные маркеры
    ctx_lower = context.lower()
    if '[add_button]' in context:
        keywords.update(['записаться', 'консультация', 'заявка', 'запись'])
    if 'http' in context:
        keywords.update(['ссылка', 'сайт', 'ресурс', 'материалы'])
    if any(x in ctx_lower for x in ['цена', 'руб', '₽', 'стоимость', 'тариф']):
        keywords.update(['цена', 'стоимость', 'тариф', 'оплата', 'сколько стоит', 'платно'])
    if any(x in ctx_lower for x in ['python', 'питон', 'пайтон']):
        keywords.update(['python', 'питон', 'пайтон', 'язык программирования'])
    if any(x in ctx_lower for x in ['группа', 'мини-группа', 'индивидуально']):
        keywords.update(['группа', 'мини-группа', 'индивидуально', 'формат'])
    
    # 6. Возвращаем максимум keywords
    return list(keywords)[:max_kw]

def update_keywords_in_db(kb_data: list, force_regenerate: bool = None) -> int:
    """
    Проверяет и обновляет keywords в main.json при старте.
    Изменяет kb_data in-place.
    
    Возвращает количество обновлённых записей.
    """
    if force_regenerate is None:
        force_regenerate = SETTINGS.get('force_regenerate', True)
    
    updated_count = 0
    
    for item in kb_data:
        should_update = force_regenerate or not item.get('keywords') or len(item.get('keywords', [])) < 5
        
        if should_update:
            new_kws = auto_generate_keywords(item['context'])
            item['keywords'] = new_kws
            updated_count += 1
    
    if updated_count > 0:
        save_json(FILES['kb'], kb_data)
        logger.info(f"✅ Авто-генерация keywords: обновлено {updated_count} записей.")
    else:
        logger.info("✅ Все keywords в порядке.")
    
    return updated_count

# ============================================================
# КЛАСС ИНДЕКСА (HYBRID SEARCH: Keywords + Context)
# ============================================================
class KBIndex:
    def __init__(self, items: list):
        self.items = items
        self.contexts = [item['context'] for item in items] if items else []
        
        # ✅ Подготовка данных для BM25
        searchable_texts = []
        for item in items:
            text = item['context']
            if item.get('keywords'):
                text += " " + " ".join(item['keywords'])
            searchable_texts.append(text)
        
        self.tokenized_contexts = [lemmatize_sentence(t).split() for t in searchable_texts] if searchable_texts else []
        self.bm25 = BM25Okapi(self.tokenized_contexts) if self.tokenized_contexts else None
        
        # Сохраняем списки ключевых слов
        self.item_keywords = [set(item.get('keywords', [])) for item in items]
        
        # Все keywords для нечеткого поиска
        self.all_keywords = []
        for item in items:
            self.all_keywords.extend(item.get('keywords', []))
        self.all_keywords = list(set(self.all_keywords))
    
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
        
        # --- Шаг 2: Бонусы за Keywords ---
        final_scores = bm25_scores.copy()
        
        for idx in range(len(self.items)):
            score_boost = 0.0
            
            for kw in self.item_keywords[idx]:
                if len(kw.split()) > 1 and kw.lower() in query_lower:
                    score_boost += 5.0
                elif kw.lower() in query_lower:
                    score_boost += 2.0
            
            # ✅ Бонус за контекст беседы
            if user_context:
                history = user_context.get('history', [])
                for hist_msg in history[-5:]:
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
    ✅ Использует приватную переменную _kb_index
    """
    global _kb_index
    
    # ✅ Сначала загружаем данные из файла
    kb_data = load_json(FILES['kb'])
    
    if not kb_data:
        logger.error("❌ База знаний пуста или файл не найден!")
        _kb_index = KBIndex([])
        return _kb_index
    
    # Обновляем keywords (передаём kb_data, функция изменит его in-place)
    update_keywords_in_db(kb_data)
    
    # Создание индекса
    _kb_index = KBIndex(kb_data)
    
    logger.info(f"✅ База знаний загружена: {len(_kb_index.items)} записей")
    return _kb_index

# ============================================================
# ПОИСК И КОНТЕКСТ
# ============================================================
def get_user_context(user_id: int) -> dict:
    """Получение или создание контекста пользователя"""
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=SETTINGS['max_history']),
            "last_activity": datetime.now(),
            "question_index_map": {}
        }
    return user_contexts[user_id]

def update_user_activity(user_id: int):
    """Обновление активности пользователя"""
    ctx = get_user_context(user_id)
    ctx["last_activity"] = datetime.now()

def save_question_for_answer(user_id: int, ans_idx: int, question: str):
    """Сохранение вопроса для конкретного ответа"""
    ctx = get_user_context(user_id)
    ctx["question_index_map"][ans_idx] = question

def get_question_for_answer(user_id: int, ans_idx: int) -> str:
    """Получение вопроса для конкретного ответа"""
    ctx = get_user_context(user_id)
    return ctx.get("question_index_map", {}).get(ans_idx, "???")

def save_message_to_history(user_id: int, message: str, is_user: bool = True):
    """Сохранение сообщения в историю"""
    ctx = get_user_context(user_id)
    prefix = "User: " if is_user else "Bot: "
    ctx["history"].append(f"{prefix}{message}")

def get_contextual_question(user_id: int, current_question: str) -> str:
    """Добавляет контекст из истории для уточняющих вопросов"""
    ctx = get_user_context(user_id)
    history = ctx.get("history", [])
    
    if not history:
        return current_question
    
    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'а скидки', 'а рассрочка', 'а документ', 
                       'и', 'тоже', 'также', 'еще', 'ещё', 'продолжи', 'далее']
    q_lower = current_question.lower()
    
    if len(q_lower) < 20 or any(marker in q_lower for marker in context_markers):
        recent_history = list(history)[-3:] if len(history) >= 3 else list(history)
        history_context = " ".join(recent_history)
        return f"{history_context} {current_question}"
    
    return current_question

def cleanup_inactive_users():
    """Очистка памяти от неактивных пользователей"""
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
    """Извлекает URL и создает структуру для кнопок"""
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