from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

# Автоматически определяем корень проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Путь к векторайзеру
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")

app = FastAPI(title="Анализ тональности отзывов (IMDB) — Production")

print(f"Загружаем векторайзер из {VECTORIZER_PATH}...")
vectorizer = joblib.load(VECTORIZER_PATH)

print("Обучаем лучшую модель локально (LogisticRegression, F1 ≈ 0.891)...")
# Загружаем данные для обучения модели
X_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "X_train.csv")).values
y_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "y_train.csv")).values.ravel()

model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X_train, y_train)

print("Модель обучена и готова к предсказаниям!")

class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "service": "Анализ тональности отзывов о фильмах",
        "model": "LogisticRegression (лучшая из MLflow экспериментов, F1 = 0.8912)",
        "status": "готов"
    }

@app.post("/predict")
def predict(review: Review):
    X_new = vectorizer.transform([review.text])
    prediction = int(model.predict(X_new)[0])
    probability = float(model.predict_proba(X_new)[0][prediction])

    sentiment = "положительный" if prediction == 1 else "отрицательный"

    return {
        "review": review.text,
        "sentiment": sentiment,
        "confidence": round(probability, 4),
        "note": "Модель обучена на IMDB датасете (50k отзывов)"
    }