import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords

# --- НАСТРОЙКИ ---
nltk.download('stopwords', quiet=True)
morph = MorphAnalyzer()
stop_ru = stopwords.words('russian')
extra_stops = ['это', 'что', 'как', 'бы', 'же', 'там', 'можно', 'было', 'есть', 'просто',
               'тоже', 'или', 'вопрос', 'ответ', 'тебе', 'меня', 'тебя', 'вы', 'вас',
               'который', 'весь', 'свой', 'мочь', 'хотеть', 'знать', 'сказать', 'говорить', 'человек']
stop_ru.extend(extra_stops)


def clean_and_lemmatize(text):
    words = re.findall(r'[а-яё]+', text.lower())
    res = []
    for w in words:
        if w not in stop_ru:
            p = morph.parse(w)
            if p.tag.POS in ['NOUN', 'VERB', 'ADJF', 'ADVB']:
                res.append(p.normal_form)
    return " ".join(res)


def get_topic_name(keywords):
    k = " ".join(keywords).lower()
    if any(w in k for w in ['путин', 'россия', 'страна', 'власть', 'закон', 'украина', 'война', 'армия']):
        return "ПОЛИТИКА / ОБЩЕСТВО"
    if any(w in k for w in
           ['деньга', 'рубль', 'купить', 'цена', 'работа', 'зарплата', 'бизнес', 'карта', 'банк', 'стоить']):
        return "ЭКОНОМИКА / ФИНАНСЫ"
    if any(w in k for w in ['ребенок', 'семья', 'жена', 'муж', 'родитель', 'жизнь', 'любовь', 'друг']):
        return "ЛИЧНАЯ ЖИЗНЬ / СЕМЬЯ"
    if any(w in k for w in ['год', 'время', 'день', 'месяц', 'дата', 'сколько', 'возраст', 'час']):
        return "ВРЕМЯ / СРОКИ"
    if any(w in k for w in ['бог', 'душа', 'смысл', 'смерть', 'вера', 'религия', 'грех']):
        return "ФИЛОСОФИЯ / РЕЛИГИЯ"
    if any(w in k for w in ['компьютер', 'телефон', 'программа', 'сайт', 'интернет', 'игра', 'видео']):
        return "ТЕХНОЛОГИИ / IT"
    if any(w in k for w in ['книга', 'фильм', 'музыка', 'читать', 'смотреть', 'автор']):
        return "КУЛЬТУРА / КИНО"
    return "ОБЩАЯ ТЕМАТИКА"


# 1. ПАРСИНГ
print("--- Шаг 1: Парсинг диалогов ---")
with open('clean_dialogues.txt', 'r', encoding='utf-8') as f:
    full_text = f.read()

pattern = re.compile(r'Вопрос:(.*?)Ответ:(.*?)(?=Вопрос:|$)', re.IGNORECASE | re.DOTALL)
matches = pattern.findall(full_text)
dialogues = [f"Вопрос: {q.strip()} Ответ: {a.strip()}" for q, a in matches if q.strip() and a.strip()]
docs = dialogues[:100000]

# 2. ВЕКТОРЫ (Кэш)
print("\n--- Шаг 2: Векторизация ---")
emb_file = 'embeddings_cache.npy'
if os.path.exists(emb_file) and len(np.load(emb_file)) == len(docs):
    embeddings = np.load(emb_file)
else:
    model = SentenceTransformer('./models')
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True)
    np.save(emb_file, embeddings)

# 3. КЛАСТЕРИЗАЦИЯ
print("\n--- Шаг 3: Кластеризация ---")
n_clusters = 15
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# 4. АНАЛИЗ И СБОР ДАННЫХ
print("\n--- Шаг 4: Анализ тематик ---")
df_res = pd.DataFrame({'text': docs, 'cluster': labels})
topic_info_list = []

for i in range(n_clusters):
    cluster_data = df_res[df_res['cluster'] == i]['text']
    if cluster_data.empty: continue

    clean_docs = [clean_and_lemmatize(t) for t in cluster_data]
    clean_docs = [d for d in clean_docs if d.strip()]

    if clean_docs:
        # Динамическая настройка min_df, чтобы избежать ошибки на маленьких кластерах
        current_min_df = min(5, len(clean_docs))
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=current_min_df)

        t_matrix = tfidf.fit_transform(clean_docs)
        importance = np.asarray(t_matrix.sum(axis=0)).flatten()
        top_idx = importance.argsort()[-10:][::-1]
        feature_names = tfidf.get_feature_names_out()
        keywords = [feature_names[idx] for idx in top_idx]

        title = get_topic_name(keywords)
        topic_info_list.append({
            'Название темы': title,
            'Ключевые слова': ", ".join(keywords),
            'Кол-во диалогов': len(cluster_data)
        })

# Сортировка по популярности
topic_info_list.sort(key=lambda x: x['Кол-во диалогов'], reverse=True)
summary_df = pd.DataFrame(topic_info_list)

# ВЫВОД ИТОГОВОЙ ТАБЛИЦЫ
print("\n" + "=" * 100)
print(f"{'№':<3} | {'КОЛ-ВО':<8} | {'НАЗВАНИЕ ТЕМЫ':<25} | {'КЛЮЧЕВЫЕ СЛОВА'}")
print("-" * 100)
for idx, row in summary_df.iterrows():
    print(f"{idx + 1:<3} | {row['Кол-во диалогов']:<8} | {row['Название темы']:<25} | {row['Ключевые слова']}")

# 5. КРУГОВАЯ ДИАГРАММА
print("\n--- Шаг 5: Построение диаграммы ---")
plt.figure(figsize=(12, 10))
# Группируем маленькие темы для красоты графика, если их слишком много
plt.pie(summary_df['Кол-во диалогов'], labels=summary_df['Название темы'],
        autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Распределение тем в датасете (%)')
plt.savefig('topic_distribution_pie.png', bbox_inches='tight')

# 6. СОХРАНЕНИЕ
summary_df.to_csv('topic_summary_table.csv', index=False, encoding='utf-8-sig')
print("\nГотово! Таблица сохранена в 'topic_summary_table.csv', график — в 'topic_distribution_pie.png'")
plt.show()
