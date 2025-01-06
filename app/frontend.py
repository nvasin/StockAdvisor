from utils.logger import logger
import datetime
from typing import Optional, List, Dict, Any

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from pydantic import BaseModel, Field, ValidationError

# ==========================
# Pydantic-модели
# ==========================

class FitRequest(BaseModel):
    """
    Модель для валидации/сериализации данных,
    которые отправляются на эндпоинт /fit.
    """
    ticker: str = Field(..., description="Идентификатор тикера")
    forecast_steps: int = Field(..., gt=0, description="Количество шагов прогноза")
    auto_select: bool = Field(..., description="Автоматический подбор гиперпараметров")
    p: Optional[int] = Field(None, description="Параметр p")
    d: Optional[int] = Field(None, description="Параметр d")
    q: Optional[int] = Field(None, description="Параметр q")
    user_data: Optional[List[Dict[str, Any]]] = Field(None, description="Пользовательские данные из CSV (при наличии)")

# ==========================
# КОНСТАНТЫ
# ==========================

# API_URL_MODELS = "http://localhost:8080/models"
# API_URL_FIT = "http://localhost:8080/fit"
# API_URL_PREDICT = "http://localhost:8080/predict"

# docker
API_URL_MODELS = "http://backend:8080/models"
API_URL_FIT = "http://backend:8080/fit"
API_URL_PREDICT = "http://backend:8080/predict"

# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def fetch_models_data() -> Dict[str, Any]:
    """
    Получение данных о моделях с эндпоинта /models
    """
    try:
        logger.info("Fetching models from %s", API_URL_MODELS)
        response = requests.get(API_URL_MODELS)
        if response.status_code == 200:
            logger.info("Данные о моделях успешно получены. Код ответа: 200")
            return response.json()
        else:
            logger.error("Не удалось получить данные о моделях. Код ответа: %d", response.status_code)
            st.error("Ошибка получения данных от API")
            return {}
    except Exception as e:
        logger.exception("Ошибка при подключении к эндпоинту /models: %s", e)
        st.error(f"Ошибка подключения к API: {e}")
        return {}

def post_fit(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Отправка запроса на эндпоинт /fit для обучения модели
    """
    try:
        logger.info("Отправка данных для обучения модели на /fit: %s", payload)
        response = requests.post(API_URL_FIT, json=payload)
        if response.status_code == 200:
            logger.info("Model fit successful. Status code: 200")
            return response.json()
        else:
            logger.error(
                "Не удалось обучить модель. Код ответа: %d, подробности: %s",
                response.status_code, 
                response.json().get('detail', 'Подробности отсутствуют')
            )
            st.error(f"Ошибка: {response.json().get('detail', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        logger.exception("Ошибка при подключении к эндпоинту /fit: %s", e)
        st.error(f"Ошибка при подключении к API: {e}")
        return None

def get_predictions(model_ids: List[str]) -> Optional[Dict[str, Any]]:
    """
    Получение прогнозов с эндпоинта /predict для списка моделей
    """
    try:
        logger.info("Запрос прогнозов с эндпоинта /predict для моделей: %s", model_ids)
        response = requests.get(API_URL_PREDICT, params={"model_id": model_ids})
        if response.status_code == 200:
            logger.info("Прогнозы успешно получены. Код ответа: 200")
            return response.json()
        else:
            logger.error(
                "Не удалось получить прогнозы. Код ответа: %d, подробности: %s",
                response.status_code, 
                response.json().get('detail', 'Подробности отсутствуют')
            )
            st.error(f"Ошибка API: {response.json().get('detail', 'Неизвестная ошибка')}")
            return None
    except Exception as e:
        logger.exception("Ошибка при подключении к эндпоинту /predict: %s", e)
        st.error(f"Ошибка подключения к API: {e}")
        return None

def render_filters(models_data: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Унифицированная функция для рендера фильтров в боковой панели.
    """
    st.sidebar.subheader("Фильтры")

    # Группы
    groups = ["Все"] + list(models_data.keys())
    selected_group = st.sidebar.selectbox("Выберите группу", groups)

    # Тикеры
    tickers = []
    if selected_group == "Все":
        for group_models in models_data.values():
            tickers.extend(group_models.keys())
    else:
        tickers = list(models_data.get(selected_group, {}).keys())
    tickers = ["Все"] + sorted(set(tickers))
    selected_tickers = st.sidebar.multiselect("Выберите тикеры", tickers, default="Все")

    # Модели
    model_ids = []
    if section in ["/predict", "/model_info"]:
        for group, tickers_data in models_data.items():
            if selected_group != "Все" and group != selected_group:
                continue
            for ticker, models in tickers_data.items():
                if "Все" not in selected_tickers and ticker not in selected_tickers:
                    continue
                for version in models["versions"]:
                    model_ids.append(version["id"])
        st.sidebar.write(f"Найдено моделей: {len(model_ids)}")
        selected_models = st.sidebar.multiselect("Выберите модели", model_ids)

    # Возврат выбранных значений
    return {
        "selected_group": selected_group,
        "selected_tickers": selected_tickers,
        "selected_models": selected_models if section in ["/predict", "/model_info"] else []
    }

def highlight_metrics(val):
    """
    Подстветка метрик.
    """
    if isinstance(val, float):
        if val < 0.2:
            return 'background-color: #007d79; color: black'
        elif 0.2 <= val < 0.5:
            return 'background-color: #d2a106; color: black'
        return 'background-color: #ba4e00; color: white'
    return ''

# ==========================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
# ==========================

st.set_page_config(page_title="StockAdvisor", layout="wide")

# Боковая панель с навигацией
st.sidebar.title("Навигация")
section = st.sidebar.radio("Выберите раздел", 
                           ["Главная", "/models", "/fit", "/predict", "/model_info","/eda"])

# ==========================
# РАЗДЕЛ "ГЛАВНАЯ"
# ==========================
if section == "Главная":
    st.title("Добро пожаловать в StockAdvisor")
    st.markdown("Используйте боковую панель для навигации.")

# ==========================
# РАЗДЕЛ "/models"
# ==========================
elif section == "/models":
    st.title("Модели StockAdvisor")
    st.markdown("Фильтруйте модели по группам и тикерам.")

    models_data = fetch_models_data()
    filters = render_filters(models_data, section="/models")

    # Применяем фильтры
    data = []
    for group, tickers_data in models_data.items():
        if filters["selected_group"] != "Все" and group != filters["selected_group"]:
            continue
        for ticker, models in tickers_data.items():
            if "Все" not in filters["selected_tickers"] and ticker not in filters["selected_tickers"]:
                continue
            for version in models["versions"]:
                parameters = version.get("parameters", {})
                metrics = version.get("metrics", {})
                forecast = version.get("forecast", {})

                data.append({
                    "Группа": group,
                    "ID модели": version["id"],
                    "Тип модели": version["name"],
                    "Тикер": ticker,
                    "Параметры": f"p={parameters.get('p', '-')}, d={parameters.get('d', '-')}, q={parameters.get('q', '-')}",
                    "MAE": round(metrics.get("mae", 0), 4),
                    "MAPE": round(metrics.get("mape", 0), 4),
                    "R2": round(metrics.get("r2", 0), 4),
                    "Последняя дата датасета": (
                        pd.to_datetime(forecast.get("last_date", "-")).strftime("%Y-%m-%d")
                        if forecast.get("last_date") else "-"
                    ),
                    "Версия": version.get("version", "-"),
                    "Сохранено": (
                        pd.to_datetime(version.get("saved_at", "-")).strftime("%Y-%m-%d %H:%M:%S")
                        if version.get("saved_at") else "-"
                    )
                })

    # Отображение таблицы
    if data:
        df = pd.DataFrame(data)
        styled_df = df.style.map(highlight_metrics, subset=["MAE", "MAPE", "R2"])
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.write("Нет данных для отображения.")

# ==========================
# РАЗДЕЛ "/fit"
# ==========================
elif section == "/fit":
    st.title("Обучение моделей (эндпоинт /fit)")

    # Форма для параметров
    st.sidebar.subheader("Параметры модели")
    ticker = st.sidebar.text_input("Введите тикер (обязательно)", "")
    forecast_steps = st.sidebar.number_input("Шаги прогноза", min_value=1, value=3, step=1)
    auto_select = st.sidebar.checkbox("Авто-подбор параметров", value=True)

    p = d = q = None
    if not auto_select:
        p = st.sidebar.number_input("Параметр p", min_value=0, value=1, step=1)
        d = st.sidebar.number_input("Параметр d", min_value=0, value=1, step=1)
        q = st.sidebar.number_input("Параметр q", min_value=0, value=1, step=1)

    # Загрузка пользовательских данных
    uploaded_file = st.file_uploader("Загрузите CSV файл с данными (опционально)", type=["csv"])
    user_data = None
    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            # Если нужно переименовать столбцы:
            # user_data.rename(columns={"Data": "Date"}, inplace=True)
            st.write("Предварительный просмотр загруженных данных:")
            st.write(user_data.head())
        except Exception as e:
            logger.exception("Error reading the uploaded file: %s", e)
            st.error(f"Ошибка загрузки файла: {e}")

    # Кнопка запуска обучения
    if st.sidebar.button("Запустить обучение"):
        if not ticker:
            st.error("Тикер обязателен для обучения.")
        else:
            # Подготавливаем payload через Pydantic-модель FitRequest
            try:
                fit_request = FitRequest(
                    ticker=ticker,
                    forecast_steps=forecast_steps,
                    auto_select=auto_select,
                    p=p,
                    d=d,
                    q=q,
                    user_data=user_data.to_dict(orient="records") if user_data is not None else None
                )
            except ValidationError as val_err:
                st.error(f"Данные не прошли валидацию: {val_err}")
                logger.error("Validation error in FitRequest: %s", val_err)
                st.stop()

            result = post_fit(fit_request.dict())
            if result:
                st.success("Обучение завершено успешно!")
                st.write(f"**ID модели**: {result.get('model_id', 'Не указано')}")
                st.write("**Результаты прогноза:**")
                st.json(result.get("predictions", "Прогнозы отсутствуют"))

# ==========================
# РАЗДЕЛ "/predict"
# ==========================
elif section == "/predict":
    st.title("Прогнозирование (эндпоинт /predict)")

    models_data = fetch_models_data()
    if not models_data:
        st.write("Нет данных о моделях для прогнозирования.")
        st.stop()

    filters = render_filters(models_data, section="/predict")

    # Использование фильтров
    if st.sidebar.button("Получить прогнозы"):
        if not filters["selected_models"]:
            st.error("Выберите хотя бы одну модель.")
        else:
            results = get_predictions(filters["selected_models"])
            if results and "models" in results:
                st.success("Прогнозы успешно получены!")
                for result in results["models"]:
                    if "error" in result:
                        st.error(f"Модель {result['model_id']} - {result['error']}")
                    else:
                        st.write(f"**Model ID**: {result['model_id']}")
                        st.write(f"**Ticker**: {result['ticker']}")
                        st.write(f"**Forecast Steps**: {result['forecast_steps']}")
                        st.write("**Predictions:**")
                        st.json(result.get("predictions", "Нет данных"))

# ==========================
# РАЗДЕЛ "/model_info"
# ==========================
elif section == "/model_info":
    st.title("Информация о модели и кривых обучения")

    models_data = fetch_models_data()
    if not models_data:
        st.write("Нет данных о моделях для отображения информации.")
        st.stop()

    st.sidebar.subheader("Фильтры")

    groups = ["Все"] + list(models_data.keys())
    selected_group = st.sidebar.selectbox("Выберите группу", groups)

    tickers = []
    if selected_group == "Все":
        for group_models in models_data.values():
            tickers.extend(group_models.keys())
    else:
        tickers = list(models_data.get(selected_group, {}).keys())
    tickers = ["Все"] + sorted(set(tickers))
    selected_ticker = st.sidebar.selectbox("Выберите тикер", tickers)

    model_ids = []
    # Логика отбора моделей
    if selected_group == "Все" and selected_ticker == "Все":
        for group_models in models_data.values():
            for ticker_models in group_models.values():
                for version in ticker_models["versions"]:
                    model_ids.append({
                        "id": version["id"], 
                        "name": version["name"], 
                        "ticker": version["ticker"]
                    })
    elif selected_group != "Все" and selected_ticker == "Все":
        for ticker_models in models_data[selected_group].values():
            for version in ticker_models["versions"]:
                model_ids.append({
                    "id": version["id"], 
                    "name": version["name"], 
                    "ticker": version["ticker"]
                })
    elif selected_ticker != "Все":
        if selected_group != "Все":
            ticker_models = models_data[selected_group].get(selected_ticker, {"versions": []})
            for version in ticker_models["versions"]:
                model_ids.append({
                    "id": version["id"], 
                    "name": version["name"], 
                    "ticker": version["ticker"]
                })
        else:
            # Если выбраны "Все группы" и конкретный тикер, ищем во всех группах
            for group_name, group_models in models_data.items():
                if selected_ticker in group_models:
                    for version in group_models[selected_ticker]["versions"]:
                        model_ids.append({
                            "id": version["id"], 
                            "name": version["name"], 
                            "ticker": version["ticker"]
                        })

    model_options = [
        f"{model['id']} ({model['name']}, {model['ticker']})" 
        for model in model_ids
    ]
    selected_model_options = st.sidebar.multiselect("Выберите модели", model_options)
    selected_model_ids = [option.split(" ")[0] for option in selected_model_options]

    if selected_model_ids:
        st.subheader("Сравнение кривых обучения")
        fig = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, selected_model_id in enumerate(selected_model_ids):
            # Ищем данные по конкретной модели
            model_info = None
            for group_name, group_models in models_data.items():
                for ticker_name, ticker_models in group_models.items():
                    for version in ticker_models["versions"]:
                        if version["id"] == selected_model_id:
                            model_info = version
                            break
                    if model_info:
                        break
                if model_info:
                    break

            if not model_info:
                logger.warning("Model with ID %s not found in data.", selected_model_id)
                continue

            forecast = model_info.get("forecast", {})
            predictions = forecast.get("predictions", [])
            if predictions:
                df_forecast = pd.DataFrame(
                    predictions, 
                    columns=["Дата", "Прогноз", "Нижняя граница", "Верхняя граница"]
                )
                df_forecast["Дата"] = pd.to_datetime(df_forecast["Дата"])

                fig.add_trace(go.Scatter(
                    x=df_forecast["Дата"],
                    y=df_forecast["Прогноз"],
                    mode='lines',
                    name=f"{model_info['id']} ({model_info['name']}, {model_info.get('ticker', '-')})",
                    line=dict(color=colors[idx % len(colors)])
                ))

        fig.update_layout(
            title="Кривые обучения выбранных моделей",
            xaxis_title="Дата",
            yaxis_title="Прогноз",
            legend_title="Модели"
        )
        st.plotly_chart(fig)

        # Детальная информация по каждой выбранной модели
        st.subheader("Детали выбранных моделей")
        for idx, selected_model_id in enumerate(selected_model_ids):
            model_info = None
            for group_name, group_models in models_data.items():
                for ticker_name, ticker_models in group_models.items():
                    for version in ticker_models["versions"]:
                        if version["id"] == selected_model_id:
                            model_info = version
                            break
                    if model_info:
                        break
                if model_info:
                    break

            if not model_info:
                continue

            parameters = model_info.get("parameters", {})
            metrics = model_info.get("metrics", {})
            forecast = model_info.get("forecast", {})

            st.markdown(
                f"<div style='background-color: {colors[idx % len(colors)]}; "
                f"padding: 10px; border-radius: 5px; color: white;'>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"### Модель: {model_info['name']} ({model_info.get('ticker', '-')})"
            )
            st.markdown(f"**ID:** {model_info['id']}")
            st.markdown(
                f"**Параметры:** p={parameters.get('p', '-')}, "
                f"d={parameters.get('d', '-')}, q={parameters.get('q', '-')}"
            )
            st.markdown(f"**Версия:** {model_info.get('version', '-')}")
            last_date = forecast.get('last_date')
            if last_date:
                last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
            else:
                last_date_str = '-'
            st.markdown(f"**Последняя дата:** {last_date_str}")

            saved_at = model_info.get("saved_at")
            if saved_at:
                saved_at_str = pd.to_datetime(saved_at).strftime('%Y-%m-%d %H:%M:%S')
            else:
                saved_at_str = '-'
            st.markdown(f"**Сохранено:** {saved_at_str}")

            st.markdown(f"**MAE:** {round(metrics.get('mae', 0), 4)}")
            st.markdown(f"**MAPE:** {round(metrics.get('mape', 0), 4)}")
            st.markdown(f"**R2:** {round(metrics.get('r2', 0), 4)}")

            if "predictions" in forecast:
                st.markdown(f"**Прогнозы:** {len(forecast['predictions'])} шагов")
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# РАЗДЕЛ "/eda"
# TODO
# ==========================
elif section == "/eda":
    st.title("EDA (Exploratory Data Analysis)")
    st.write("Раздел для визуализации данных. В разработке...")
