from prometheus_client import Counter, Histogram, start_http_server
import time


# Метрики
REQUESTS = Counter('requests_total', 'Total requests')
LATENCY = Histogram('request_latency_seconds', 'Request latency')

# Запускаем Prometheus-эндпоинт на порту 8001
start_http_server(8001)


@app.post("/predict")
def predict(review: Review):
    REQUESTS.inc()
    with LATENCY.time():
        start = time.time()
        # ... твой код предсказания ...
        end = time.time()

    return {...}