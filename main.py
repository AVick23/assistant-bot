import json
import re
import numpy as np
import warnings
from typing import Dict, List, Set, Optional
import math
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Загрузка переменных окружения из .env файла
load_dotenv()

# Легковесные зависимости для обработки русского языка
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Константы
ADMIN_USER_ID = 1373472999
CONSULTATIONS_FILE = "consultations.json"
CALENDAR_URL = "https://calendar.app.google/ThpteAc5uqhxqnUA9"

# Инициализация морфологического анализатора для русского языка
morph = pymorphy2.MorphAnalyzer()

# Расширенный список стоп-слов для русского языка
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

# Словарь синонимов для улучшения поиска
SYNONYMS = {
    'стоимость': ['цена', 'тариф', 'плата', 'расценка'],
    'курс': ['обучение', 'программа', 'тренинг'],
    'преподаватель': ['учитель', 'репетитор', 'тренер', 'лектор'],
    'занятие': ['урок', 'лекция', 'пара', 'встреча'],
    'группа': ['команда', 'коллектив'],
    'метод': ['подход', 'техника', 'стратегия'],
    'домашка': ['задание', 'дз', 'практика'],
    'бот': ['чат-бот', 'ассистент', 'помощник'],
    'python': ['питон', 'пайтон'],
    'программирование': ['кодинг', 'разработка'],
    'вопрос': ['запрос', 'проблема', 'тема'],
    'ответ': ['решение', 'отклик'],
    'начать': ['стартовать', 'приступить'],
    'записаться': ['зарегистрироваться', 'подписаться'],
    'сложный': ['трудный', 'замысловатый', 'запутанный'],
    'легкий': ['простой', 'нетрудный'],
    'быстро': ['скорость', 'оперативно', 'в срок'],
    'долго': ['медленно', 'затянуто'],
    'качество': ['уровень', 'стандарт'],
    'консультация': ['встреча', 'совет', 'помощь'],
    'доступ': ['получение', 'возможность'],
    'материалы': ['уроки', 'лекции', 'ресурсы'],
    'поддержка': ['помощь', 'сопровождение']
}

def preprocess_question(question: str) -> str:
    """Удаляет вводные конструкции из вопроса"""
    patterns = [
        r'^а если\s+',
        r'^что если\s+',
        r'^что будет если\s+',
        r'^можно ли\s+',
        r'^а что если\s+',
        r'^если я\s+',
        r'^а\s+',
        r'^ну\s+',
        r'^скажи\s+',
        r'^расскажи\s+',
        r'^объясни\s+'
    ]
    
    cleaned = question.lower()
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned.strip()

def expand_with_synonyms(keywords: Set[str]) -> Set[str]:
    """Расширение набора ключевых слов синонимами"""
    expanded = set(keywords)
    for word in keywords:
        for base, synonyms in SYNONYMS.items():
            if word == base or any(word == syn for syn in synonyms):
                expanded.update([base] + synonyms)
    return expanded

def load_knowledge_base(file_path: str) -> list:
    """Загрузка базы знаний из JSON-файла"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_text(text: str) -> str:
    """Очистка текста от специальных символов"""
    # Удаляем URL, email и т.д.
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Основная очистка
    return re.sub(r'[^\w\s]', ' ', text.lower().strip())

def lemmatize_word(word: str) -> str:
    """Лемматизация одного слова для русского языка с кэшированием"""
    # Кэширование для ускорения
    if not hasattr(lemmatize_word, 'cache'):
        lemmatize_word.cache = {}
    
    if word in lemmatize_word.cache:
        return lemmatize_word.cache[word]
    
    parsed = morph.parse(word)[0]
    lemma = parsed.normal_form
    lemmatize_word.cache[word] = lemma
    return lemma

def lemmatize_sentence(text: str) -> str:
    """Лемматизация всего предложения"""
    # Удаляем знаки вопроса и другие лишние символы
    text = re.sub(r'[?!.]', '', text)
    words = preprocess_text(text).split()
    lemmas = [lemmatize_word(word) for word in words if not is_stop_word(word) and len(word) > 2]
    return " ".join(lemmas)

def is_stop_word(word: str) -> bool:
    """Проверка, является ли слово стоп-словом"""
    return word.lower() in RUSSIAN_STOPWORDS

def extract_keywords(text: str, use_synonyms: bool = True) -> set:
    """Извлечение ключевых слов с лемматизацией"""
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    
    keywords = {
        lemmatize_word(word) for word in words
        if len(word) > 2 and not is_stop_word(word)
    }
    
    # Расширение синонимами
    if use_synonyms:
        keywords = expand_with_synonyms(keywords)
    
    return keywords

def extract_entities(text: str) -> dict:
    """Простое извлечение сущностей (числа, даты, имена)"""
    entities = {
        'numbers': re.findall(r'\d+', text),
        'money': re.findall(r'\d+\s*(?:руб|р|рублей|долларов|usd|eur)', text, re.IGNORECASE),
        'timeframes': re.findall(r'\d+\s*(?:час|минут|дней|недел|месяц|год)', text, re.IGNORECASE)
    }
    return entities

def calculate_keyword_match_score(user_keywords: Set[str], item_keywords: Set[str], 
                                 user_question: str, original_keywords: List[str]) -> float:
    """Расчет оценки совпадения по ключевым словам с комплексным подходом"""
    # Базовое пересечение лемм
    common_keywords = user_keywords.intersection(item_keywords)
    base_score = len(common_keywords) * 2
    
    # Бонус за точные совпадения фраз
    question_lower = preprocess_text(user_question)
    phrase_bonus = 0
    
    for orig_keyword in original_keywords:
        keyword_lower = preprocess_text(orig_keyword)
        if keyword_lower in question_lower:
            # Бонус зависит от длины фразы и уникальности
            phrase_bonus += len(keyword_lower.split()) * 3
    
    # Дополнительные бонусы за совпадение контекста
    context_bonus = 0
    question_lemmas = set(preprocess_text(user_question).split())
    
    # Если есть числа в вопросе и в ключевых словах
    question_numbers = set(re.findall(r'\d+', user_question))
    keyword_numbers = set()
    for kw in original_keywords:
        keyword_numbers.update(re.findall(r'\d+', kw))
    
    if question_numbers and keyword_numbers and question_numbers.intersection(keyword_numbers):
        context_bonus += 5
    
    # Суммируем все компоненты
    total_score = base_score + phrase_bonus + context_bonus
    
    return total_score

class KBIndex:
    """Класс для индексации и поиска по базе знаний"""
    def __init__(self):
        self.items = []
        self.contexts = []
        self.tfidf_vectorizer = None
        self.tfidf_labeled_matrix = None
        self.raw_tfidf_vectorizer = None
        self.tfidf_raw_matrix = None
        self.last_update = 0
    
    def build_tfidf_index(self, contexts: List[str]):
        """Построение TF-IDF индекса для полнотекстового поиска"""
        # Для лемматизированных текстов
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 3),
            max_features=3000  # Уменьшено для экономии памяти
        )
        
        lemmatized_contexts = [lemmatize_sentence(ctx) for ctx in contexts]
        self.tfidf_labeled_matrix = self.tfidf_vectorizer.fit_transform(lemmatized_contexts)
        
        # Для необработанных текстов (для дополнительного поиска)
        self.raw_tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(RUSSIAN_STOPWORDS),
            ngram_range=(1, 2),
            max_features=2000
        )
        self.tfidf_raw_matrix = self.raw_tfidf_vectorizer.fit_transform(contexts)
    
    def keyword_search(self, user_question: str, top_k: int = 3) -> List[dict]:
        """Поиск по ключевым словам с ранжированием"""
        user_keywords = extract_keywords(user_question)
        print(f"🔑 Извлеченные ключевые слова из '{user_question}': {user_keywords}")
        
        if not user_keywords:
            return []
        
        scored_items = []
        for idx, item in enumerate(self.items):
            score = calculate_keyword_match_score(
                user_keywords, 
                item["keywords"], 
                user_question,
                item["original_keywords"]
            )
            
            if score > 0:
                scored_items.append({
                    "context": item["context"],
                    "score": score,
                    "index": idx
                })
        
        # Сортируем по оценке и возвращаем топ-K
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:top_k]
    
    def fulltext_search(self, query: str, top_k: int = 3) -> List[dict]:
        """Полнотекстовый поиск с использованием TF-IDF"""
        if self.tfidf_vectorizer is None or self.tfidf_labeled_matrix is None:
            return []
        
        results = []
        
        try:
            # Поиск по лемматизированному тексту
            query_lemma = lemmatize_sentence(query)
            query_vec = self.tfidf_vectorizer.transform([query_lemma])
            labeled_similarities = cosine_similarity(query_vec, self.tfidf_labeled_matrix)[0]
            
            # Поиск по необработанному тексту
            raw_query_vec = self.raw_tfidf_vectorizer.transform([query])
            raw_similarities = cosine_similarity(raw_query_vec, self.tfidf_raw_matrix)[0]
            
            # Комбинируем результаты
            combined_similarities = 0.7 * labeled_similarities + 0.3 * raw_similarities
            
            # Получаем топ-K результатов
            top_indices = np.argsort(combined_similarities)[::-1][:top_k]
            
            for idx in top_indices:
                score = combined_similarities[idx]
                if score > 0.15:  # Порог для TF-IDF
                    results.append({
                        "context": self.contexts[idx],
                        "score": float(score),
                        "index": int(idx)
                    })
        except Exception as e:
            print(f"Ошибка при полнотекстовом поиске: {str(e)}")
            # Возвращаем пустой результат в случае ошибки
        
        return results

def preprocess_knowledge_base(knowledge_base: list) -> KBIndex:
    """Предобработка базы знаний с использованием класса индексации"""
    kb_index = KBIndex()
    processed_items = []
    
    contexts = [item["context"] for item in knowledge_base]
    
    for i, item in enumerate(knowledge_base):
        # Обработка ключевых слов из базы
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
    
    # Заполняем индекс
    kb_index.items = processed_items
    kb_index.contexts = contexts
    
    # Построение TF-IDF индекса
    kb_index.build_tfidf_index(contexts)
    
    kb_index.last_update = time.time()
    return kb_index

def find_best_match(user_question: str, kb_index: KBIndex) -> str:
    """Улучшенный гибридный поиск лучшего совпадения в базе знаний"""
    # Предобработка вопроса для удаления вводных конструкций
    cleaned_question = preprocess_question(user_question)
    print(f"Оригинальный вопрос: '{user_question}'")
    print(f"Очищенный вопрос: '{cleaned_question}'")
    
    # Извлекаем сущности из вопроса
    entities = extract_entities(user_question)
    
    # Поиск по ключевым словам для очищенного вопроса
    keyword_results = kb_index.keyword_search(cleaned_question, top_k=5)
    
    # Поиск по полному тексту для очищенного вопроса
    fulltext_results = kb_index.fulltext_search(cleaned_question, top_k=5)
    
    # Если нет результатов для очищенного вопроса - пробуем исходный вопрос
    if not keyword_results and not fulltext_results:
        print("Нет результатов для очищенного вопроса, пробуем исходный вопрос")
        keyword_results = kb_index.keyword_search(user_question, top_k=5)
        fulltext_results = kb_index.fulltext_search(user_question, top_k=5)
    
    # Объединяем результаты с весами
    combined_results = {}
    
    # Добавляем результаты по ключевым словам (вес 0.6)
    for res in keyword_results:
        idx = res["index"]
        combined_results.setdefault(idx, 0)
        combined_results[idx] += res["score"] * 0.6
    
    # Добавляем результаты полнотекстового поиска (вес 0.4)
    for res in fulltext_results:
        idx = res["index"]
        combined_results.setdefault(idx, 0)
        combined_results[idx] += res["score"] * 50 * 0.4
    
    # Сортируем по общей оценке
    if combined_results:
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        best_idx, best_score = sorted_results[0]
        
        # Понижаем порог для принятия ответа
        if best_score > 1.5:  # Было 3.0
            print(f"Найден ответ с оценкой {best_score} для вопроса '{cleaned_question}'")
            return kb_index.items[best_idx]["context"]
    
    # Если ничего хорошего не найдено, используем лучший результат из полнотекстового поиска
    if fulltext_results:
        best_fulltext = fulltext_results[0]
        if best_fulltext["score"] > 0.2:  # Было 0.3
            print(f"Используем полнотекстовый результат с оценкой {best_fulltext['score']}")
            return best_fulltext["context"]
    
    # Если ничего не найдено - пробуем поиск по отдельным ключевым словам
    fallback_keywords = extract_keywords(cleaned_question, use_synonyms=False)
    if fallback_keywords:
        print(f"Попытка поиска по ключевым словам: {fallback_keywords}")
        fallback_results = kb_index.keyword_search(" ".join(fallback_keywords), top_k=3)
        if fallback_results and fallback_results[0]["score"] > 0:
            return fallback_results[0]["context"]
    
    # Если ничего не найдено
    return "К сожалению, я не нашел ответа на ваш вопрос в своей базе знаний. Попробуйте задать вопрос другими словами или уточнить детали."

# Глобальные переменные для хранения индекса и контекста пользователей
kb_index = None
user_contexts = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    welcome_message = (
        "Привет! Я ваш учебный помощник. "
        "Задайте любой вопрос по курсу, и я постараюсь найти ответ.\n\n"
        "ℹ️ Используется улучшенный поиск по ключевым словам и текстовому анализу."
    )
    await update.message.reply_text(welcome_message)

async def consultation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатия кнопки записи на консультацию"""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Формируем данные пользователя
    user_data = {
        "user_id": user.id,
        "username": user.username or "Нет username",
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "timestamp": timestamp
    }
    
    # Загружаем существующие данные или создаем новый список
    consultations = []
    if os.path.exists(CONSULTATIONS_FILE):
        with open(CONSULTATIONS_FILE, "r", encoding="utf-8") as f:
            try:
                consultations = json.load(f)
            except json.JSONDecodeError:
                consultations = []
    
    # Добавляем новую запись
    consultations.append(user_data)
    
    # Сохраняем обновленные данные
    with open(CONSULTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(consultations, f, ensure_ascii=False, indent=4)
    
    # Создаем кнопки с ссылкой на календарь
    keyboard = [
        [InlineKeyboardButton("📅 Перейти к расписанию", url=CALENDAR_URL)],
        [InlineKeyboardButton("📱 Написать в Telegram", url="https://t.me/AVick23")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Отправляем подтверждение с дополнительными вариантами связи
    await query.edit_message_text(
        text="✅ Ваша заявка успешно сохранена!\n\n"
             "Вы можете:\n"
             "1. 🔗 <b>Выбрать удобное время сами</b> через Google Календарь\n"
             "2. 📱 <b>Написать мне напрямую</b> в Telegram для согласования времени\n\n"
             "Я также свяжусь с вами в ближайшее время для подтверждения деталей.",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def clear_list_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатия кнопки очистки списка заявок"""
    query = update.callback_query
    await query.answer()
    
    # Очищаем файл заявок
    with open(CONSULTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)
    
    # Отправляем подтверждение
    await query.edit_message_text(text="✅ Список заявок успешно очищен!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений"""
    user_id = update.effective_user.id
    user_question = update.message.text.strip().lower()
    
    # Обработка команды "заявки" только для администратора
    if user_id == ADMIN_USER_ID and user_question == "заявки":
        if not os.path.exists(CONSULTATIONS_FILE) or os.path.getsize(CONSULTATIONS_FILE) == 0:
            await update.message.reply_text("📋 Список заявок пуст.")
            return
        
        try:
            with open(CONSULTATIONS_FILE, "r", encoding="utf-8") as f:
                consultations = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            consultations = []
        
        if not consultations:
            await update.message.reply_text("📋 Список заявок пуст.")
            return
        
        # Формируем сообщение со списком заявок
        message = "📋 Список заявок:\n\n"
        for idx, consult in enumerate(consultations, 1):
            username = consult.get('username', 'Нет username')
            first_name = consult.get('first_name', '')
            last_name = consult.get('last_name', '')
            timestamp = consult.get('timestamp', '')
            
            message += f"{idx}. {first_name} {last_name}\n"
            message += f"   👤 @{username}\n"
            message += f"   ⏰ {timestamp}\n\n"
        
        # Создаем кнопку очистки списка
        keyboard = [[InlineKeyboardButton("Очистить список", callback_data="clear_list")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(message, reply_markup=reply_markup)
        return
    
    # Инициализация контекста для нового пользователя
    if user_id not in user_contexts:
        user_contexts[user_id] = {"last_answer": None}
    
    # Обработка коротких ответов
    short_answers = ['да', 'конечно', 'ага', 'угу', 'еще', 'больше', 'расскажи подробнее', 'как?', 'почему?']
    if user_question in short_answers:
        last_answer = user_contexts[user_id]["last_answer"]
        if last_answer:
            # Проверяем, нужно ли добавить кнопку для консультации
            if "[add_button]" in last_answer:
                clean_answer = last_answer.replace("[add_button]", "").strip()
                keyboard = [[InlineKeyboardButton("Записаться на бесплатную консультацию", callback_data="consultation")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(clean_answer, reply_markup=reply_markup)
            else:
                await update.message.reply_text(last_answer)
            return
    
    # Поиск ответа
    answer = find_best_match(update.message.text, kb_index)
    
    # Сохранение контекста (очищенный ответ без маркера)
    clean_answer = answer.replace("[add_button]", "").strip()
    user_contexts[user_id]["last_answer"] = clean_answer
    
    # Проверяем, нужно ли добавить кнопку для консультации
    if "[add_button]" in answer:
        keyboard = [
            [InlineKeyboardButton("📅 Записаться через Google Календарь", url=CALENDAR_URL)],
            [InlineKeyboardButton("📝 Оставить заявку для обратной связи", callback_data="consultation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            clean_answer + "\n\n💡 Вы можете выбрать удобное время сами через календарь или оставить заявку и я свяжусь с вами для согласования времени.",
            reply_markup=reply_markup,
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(clean_answer)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка ошибок"""
    print(f"Произошла ошибка: {context.error}")
    if update and hasattr(update, 'message'):
        await update.message.reply_text(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте задать вопрос еще раз или переформулируйте его."
        )

def main() -> None:
    """Основная функция запуска бота"""
    global kb_index
    
    # Загрузка токена из .env файла
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("Токен бота не найден в .env файле. Укажите BOT_TOKEN=ваш_токен")
    
    # Загрузка базы знаний
    try:
        kb = load_knowledge_base('main.json')
        kb_index = preprocess_knowledge_base(kb)
        print("База знаний успешно загружена и обработана")
    except Exception as e:
        print(f"Ошибка при загрузке базы знаний: {str(e)}")
        raise
    
    # Создание приложения
    application = Application.builder().token(token).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(consultation_callback, pattern="consultation"))
    application.add_handler(CallbackQueryHandler(clear_list_callback, pattern="clear_list"))
    application.add_error_handler(error_handler)
    
    # Запуск бота
    print("Бот запущен и готов к работе!")
    application.run_polling()

if __name__ == "__main__":
    main()