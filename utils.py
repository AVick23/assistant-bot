import json
import re
import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
from rank_bm25 import BM25Okapi
import numpy as np
from config import FILES, RUSSIAN_STOPWORDS, morph, logger, SETTINGS

_kb_index = None
user_contexts: Dict[int, dict] = {}

def get_kb_index() -> 'KBIndex':
    return _kb_index

def load_json(file_path: str) -> list:
    if not os.path.exists(file_path): return []
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

def preprocess_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    if not hasattr(lemmatize_word, 'cache'): lemmatize_word.cache = {}
    if word in lemmatize_word.cache: return lemmatize_word.cache[word]
    try:
        parsed = morph.parse(word)[0]
        lemma = parsed.normal_form
        lemmatize_word.cache[word] = lemma
        return lemma
    except: return word

def lemmatize_sentence(text: str) -> str:
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(w) for w in words if w not in RUSSIAN_STOPWORDS and len(w) > 2]
    return " ".join(lemmas)

# ============================================================
# КЛАСС ИНДЕКСА (Strict JSON Keywords)
# ============================================================
class KBIndex:
    def __init__(self, items: list):
        self.items = items
        self.contexts = [item['context'] for item in items] if items else []
        
        # Подготовка для BM25 (используем context + keywords для индексации)
        searchable_texts = []
        for item in items:
            text = item['context']
            if item.get('keywords'):
                text += " " + " ".join(item['keywords'])
            searchable_texts.append(text)
        
        self.tokenized_contexts = [lemmatize_sentence(t).split() for t in searchable_texts]
        self.bm25 = BM25Okapi(self.tokenized_contexts) if self.tokenized_contexts else None
        
        # Сохраняем исходные keywords для точного матчинга
        self.item_keywords = [item.get('keywords', []) for item in items]

    def search(self, query: str, top_k: int = 5, user_context: Optional[dict] = None) -> List[dict]:
        if not self.items or not self.bm25: return []

        query_lemmas = lemmatize_sentence(query).split()
        query_lower_raw = preprocess_text(query) # "кто ты"
        
        # 1. Базовый скоринг BM25
        bm25_scores = self.bm25.get_scores(query_lemmas)
        final_scores = bm25_scores.copy()
        
        # 2. Бонусы за Keywords (Исправленная логика)
        for idx in range(len(self.items)):
            score_boost = 0.0
            
            # Проверяем каждый keyword из JSON
            for kw in self.item_keywords[idx]:
                kw_lower = kw.lower()
                
                # Приоритет 1: Фраза целиком (даже если в ней есть стоп-слова)
                # "кто ты" in "кто ты такой" -> True
                if len(kw_lower.split()) > 1:
                    if kw_lower in query_lower_raw:
                        score_boost += 10.0 # Большой бонус за точную фразу
                else:
                    # Приоритет 2: Одно слово
                    if kw_lower in query_lower_raw.split():
                        score_boost += 3.0
            
            # Контекст беседы
            if user_context:
                history = list(user_context.get('history', []))[-5:]
                for hist_msg in history:
                    hist_lemmas = set(lemmatize_sentence(hist_msg).split())
                    overlap = len(hist_lemmas & set(query_lemmas))
                    if overlap > 0: score_boost += overlap * 0.2

            final_scores[idx] += score_boost

        # 3. Сортировка
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = final_scores[idx]
            # Отсекаем совсем низкие оценки
            if score > SETTINGS.get('min_score_threshold', 0.5):
                results.append({
                    "index": int(idx),
                    "score": float(score),
                    "context": self.contexts[idx],
                    "topic": self.items[idx].get('keywords', ['Тема'])[0]
                })
        return results

    def is_valid_index(self, idx: int) -> bool:
        return 0 <= idx < len(self.items)

def initialize_kb() -> KBIndex:
    global _kb_index
    kb_data = load_json(FILES['kb'])
    
    if not kb_data:
        logger.error("❌ База знаний пуста!")
        _kb_index = KBIndex([])
        return _kb_index
    
    # ✅ Никакой перезаписи! Просто создаем индекс.
    _kb_index = KBIndex(kb_data)
    logger.info(f"✅ База знаний загружена: {len(_kb_index.items)} записей (JSON Strict Mode)")
    return _kb_index

# ... (остальные функции get_user_context, update_user_activity и т.д. оставляем как были) ...
def get_user_context(user_id: int) -> dict:
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "history": deque(maxlen=SETTINGS['max_history']),
            "last_activity": datetime.now(),
            "question_index_map": {}
        }
    return user_contexts[user_id]

def update_user_activity(user_id: int):
    get_user_context(user_id)["last_activity"] = datetime.now()

def save_question_for_answer(user_id: int, ans_idx: int, question: str):
    get_user_context(user_id)["question_index_map"][ans_idx] = question

def get_question_for_answer(user_id: int, ans_idx: int) -> str:
    return get_user_context(user_id).get("question_index_map", {}).get(ans_idx, "???")

def save_message_to_history(user_id: int, message: str, is_user: bool = True):
    ctx = get_user_context(user_id)
    prefix = "User: " if is_user else "Bot: "
    ctx["history"].append(f"{prefix}{message}")

def get_contextual_question(user_id: int, current_question: str) -> str:
    ctx = get_user_context(user_id)
    history = list(ctx.get("history", []))
    if not history: return current_question
    
    context_markers = ['а', 'а есть', 'а как', 'а сколько', 'тоже', 'еще']
    q_lower = current_question.lower()
    if len(q_lower) < 20 or any(m in q_lower for m in context_markers):
        return f"{' '.join(history[-3:])} {current_question}"
    return current_question

def cleanup_inactive_users():
    now = datetime.now()
    to_delete = [uid for uid, ctx in user_contexts.items() if now - ctx.get("last_activity", now) > timedelta(hours=SETTINGS['inactivity_hours'])]
    for uid in to_delete: del user_contexts[uid]

def extract_links_and_buttons(text: str) -> Tuple[str, List[List[dict]]]:
    buttons = []
    url_pattern = r'(https?://[^\s<]+)'
    urls = re.findall(url_pattern, text)
    if urls:
        for raw_url in set(urls):
            clean_url = raw_url.replace("[add_button]", "").strip('.,;:!?() "\'[]{}')
            if not clean_url: continue
            label = "🔗 Ссылка"
            if "roadmap" in clean_url.lower(): label = "🗺 Карта"
            elif "Business-card" in clean_url: label = "👤 Преподаватель"
            elif "calendar" in clean_url.lower(): label = "📅 Календарь"
            buttons.append([{"text": label, "url": clean_url}])
        clean_text = re.sub(url_pattern, '', text).strip()
        return clean_text, buttons
    return text, []