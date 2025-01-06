FROM python:3.9

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1

# Устанавливаем системные зависимости (для отладки и работы)
RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools curl iproute2 && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY ../requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Копируем файлы приложения
COPY ../ /StockAdvisor/
ENV PYTHONPATH=/StockAdvisor/app



# Создаем папку для логов
RUN mkdir -p /logs

# Указываем порт для FastAPI
EXPOSE 8080

# Команда для запуска приложения
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]