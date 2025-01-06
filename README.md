# StockAdvisor
# Тема №17: Биржевая и внебиржевая торговля, оптимизация портфеля

## 📈 Оптимизация и диверсификация портфеля ценных бумаг

*Проект посвящен разработке Telegram-бота, который анализирует портфель пользователя, мониторит новостной фон отрасли и предоставляет рекомендации по покупке или продаже ценных бумаг с использованием алгоритмов биржевой торговли и методов искусственного интеллекта.*

---

## 👥 Состав команды

| Участник            | Telegram                                     | GitHub                                          |
|---------------------|----------------------------------------------|-------------------------------------------------|
| **Никита Васин**    | [@ni_vasin](https://t.me/ni_vasin)          | [nvasin](https://github.com/nvasin)             |
| **Никита Саранов**  | [@OUR_ELEPHANT_SB](https://t.me/OUR_ELEPHANT_SB) | [Rassssssss](https://github.com/Rassssssss)             |
| **Алексей Беляев**  | [@AlexeyBelyaev1](https://t.me/AlexeyBelyaev1) | *GitHub*                               |
| **Динар Заяхов**    | [@DinarZayahov](https://t.me/DinarZayahov)  | [DinarZayahov](https://github.com/DinarZayahov) |

---

## 📝 Кураторы

- **Куратор:** Aleksandra Kovaleva - [@ak0va](https://t.me/ak0va)
- **Мета-куратор:** Evgeny - [@evg_pat](https://t.me/evg_pat), [evgpat](https://github.com/evgpat)

---
# StockAdvisor

StockAdvisor — это приложение для анализа и прогнозирования временных рядов, которое позволяет:
- Автоматически обрабатывать данные о ценах акций.
- Обучать и сохранять модели временных рядов (ARIMA, SES, Linear Regression).
- Выполнять прогнозирование на основе обученных моделей.
- Просматривать прогнозы и метрики через удобный веб-интерфейс.

---

## Структура проекта

- **/app**:
  - Содержит frontend и backend компоненты:
    - `backend.py`: FastAPI endpoints для управления моделями и прогнозами.
    - `frontend.py`: Приложение Streamlit для взаимодействия с пользователем.
    - Вспомогательные модули для обработки данных, логирования и реализации моделей.

- **/checkpoints**:
  - Содержит старые файлы и наработки из предыдущих этапов проекта.

- **/datasets**:
  - Директория для хранения исходных и обработанных данных.

- **/docker**:
  - Конфигурация Docker для сборки и запуска контейнеров:
    - `backend.Dockerfile`, `frontend.Dockerfile`, `docker-compose.yaml`.

- **/log**:
  - Файлы логов для мониторинга работы приложения (`StockAdvisor.log`).

---

## Установка и запуск

### Требования
1. Установленые Docker и Docker Compose.

### Сборка и запуск контейнеров
1. Клонируйте репозиторий:
   ```bash
   git clone <URL_репозитория>
   cd StockAdvisor
   ```
2. Выполните сборку и запуск контейнеров:
   ```bash
   docker-compose up --build
   ```
3. После запуска:
   - Backend доступен по адресу [http://localhost:8080](http://localhost:8080).
   - Frontend доступен по адресу [http://localhost:8501](http://localhost:8501).

## Использование

### Backend API
1. **Просмотр всех моделей**:
   - Эндпоинт: `/models` (GET)
   - Возвращает список всех обученных моделей.
2. **Обучение модели**:
   - Эндпоинт: `/fit` (POST)
   - Пример запроса:
     ```json
        {
            "ticker": "ALRS",
            "forecast_steps": 5,
            "auto_select": false,
            "p": 3,
            "d": 1,
            "q": 3,
            "user_data": [
                {
                    "Date": "2024-10-26",
                    "Price": 145.2
                },
                {
                    "Date": "2024-10-26",
                    "Price": 146.0
                },
                {
                    "Date": "2023-01-03",
                    "Price": 147.5
                },
                {
                    "Date": "2023-01-04",
                    "Price": 148.1
                }
            ]
        }
     ```
   - Обучает модель на основе данных.
3. **Получение прогнозов**:
   - Эндпоинт: `/predict` (GET)
   - Параметры: `model_id` — список ID моделей.

### Frontend
- Интерфейс Streamlit позволяет:
  - Загружать данные.
  - Обучать модели через форму.
  - Просматривать прогнозы и метрики.

---

## Разработка

### Основные зависимости
Убедитесь, что установлены зависимости из файла `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Запуск локально
1. Backend:
   ```bash
   uvicorn backend:app --reload --host 0.0.0.0 --port 8080
   ```
2. Frontend:
   ```bash
   streamlit run frontend.py
   ```

---

## Логи и данные
- Логи: `./logs/StockAdvisor.log`
- Данные: `./datasets`
