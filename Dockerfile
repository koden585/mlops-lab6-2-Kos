# Базовый образ Python
FROM python:3.12-slim

# Рабочая папка внутри контейнера
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код и данные
COPY src/ src/
COPY models/vectorizer.pkl models/vectorizer.pkl
COPY data/processed/ data/processed/

# Экспонируем порт 8000
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]