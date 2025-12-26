import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

print("Обучение моделей для анализа тональности отзывов")

# Загружаем данные
print("Загружаем данные из DVC...")
X_train = pd.read_csv('../data/processed/X_train.csv').values
X_test = pd.read_csv('../data/processed/X_test.csv').values
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

# Настраиваем эксперимент
mlflow.set_experiment("imdb_sentiment_analysis_v1")

# 1. Логистическая регрессия
with mlflow.start_run(run_name="LogisticRegression"):
    model = LogisticRegression(max_iter=500, solver='liblinear')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print(f"-> LogisticRegression: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

# 2. Случайный лес
with mlflow.start_run(run_name="RandomForest_100"):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print(f"-> RandomForest: Accuracy = {acc:.4f}, F1 = {f1:.4f}")

print("\nОбучение завершено!")
print("Запуск в терминале: mlflow ui")
print("Открыть: http://localhost:5000")