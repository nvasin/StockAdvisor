FROM python:3.9

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1

# Устанавливаем системные зависимости (при необходимости для Streamlit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    net-tools curl iproute2 && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY ../requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
#COPY ../ /StockAdvisor/
#ENV PYTHONPATH=/StockAdvisor/
COPY ../app/frontend.py /app/

# Указываем порт для Streamlit
EXPOSE 8501

# Команда для запуска Streamlit
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
