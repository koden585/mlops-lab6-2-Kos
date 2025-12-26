import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

print("Подготовка данных для анализа тональности")

print("Загружаем датасет...")
df = pd.read_csv('../data/raw/imdb_reviews.csv')
print(f"Загружено отзывов: {len(df)}")

print("Преобразуем метки в числа...")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("Векторизуем текст (TF-IDF, подождите)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

print("Делим на train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Сохраняем файлы...")
pd.DataFrame(X_train.toarray()).to_csv('../data/processed/X_train.csv', index=False)
pd.DataFrame(X_test.toarray()).to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)
joblib.dump(vectorizer, '../models/vectorizer.pkl')

print("ГОТОВО! Данные готовы для обучения модели")