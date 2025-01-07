from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import uuid
from statsmodels.tsa.arima.model import ARIMA
from utils.logger import logger
from utils.model_utils import compute_metrics
from utils.model_storage import TRAINED_ARIMA_MODELS, save_model

def arima_model(
    data: pd.DataFrame,
    ticker: str,
    save_to: str = "auto_initialized",
    model_id: Optional[str] = None,
    auto_select: bool = True,
    p: Optional[int] = None,
    d: Optional[int] = None,
    q: Optional[int] = None,
    forecast_steps: int = 0,
    overwrite: bool = False,
) -> Tuple[List[Tuple[pd.Timestamp, float, float, float]], str, str]:
    """
    Универсальная функция для работы с ARIMA.
    - Может переобучать существующую модель или создавать новую.
    - Поддерживает сохранение моделей в указанные группы.
    - Сохраняет модель и метрики единовременно, чтобы избежать дублей.

    Аргументы:
    - data: pd.DataFrame с 'Price' в индексе DateTime.
    - ticker: Название тикера для сохранения модели.
    - save_to: Группа для сохранения модели (напр. 'auto_initialized').
    - model_id: ID существующей модели (для переобучения). Если None, создаётся новая.
    - auto_select: Если True, автоматически подбирает (p, d, q).
    - p, d, q: Если auto_select=False, используйте эти параметры.
    - forecast_steps: Количество шагов прогноза; если 0, прогноз не выполняется.
    - overwrite: Если True, перезаписывает существующую модель с model_id.

    Возвращает:
    - Tuple: 
        - List[Tuple[pd.Timestamp, float, float, float]]: Прогноз [(дата, предсказание, нижняя граница, верхняя граница)].
        - str: ID модели.
        - str: Имя модели.
    """
    logger.info(f"Инициализация ARIMA модели для тикера {ticker}...")

    model_name = 'ARIMA'

    # Проверка/приведение индекса к DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.debug("Преобразование индекса DataFrame в DatetimeIndex.")
        data = data.set_index('Date')
        data.index = pd.to_datetime(data.index)

    series_for_fit = data['Price']

    # Переобучение существующей модели
    if model_id:
        logger.info(f"Переобучение существующей модели с ID {model_id} в группе {save_to}.")
        group = TRAINED_ARIMA_MODELS.get(save_to, {})
        ticker_info = group.get(ticker, {})
        versions = ticker_info.get("versions", [])

        model_info = next((version for version in versions if version["id"] == model_id), None)
        if model_info is None:
            logger.error(f"Модель с ID {model_id} не найдена в группе {save_to} для тикера {ticker}.")
            raise ValueError(f"Модель с ID {model_id} не найдена.")

        # Подтягиваем старые параметры
        auto_select = model_info["parameters"]["auto_select"]
        p = model_info["parameters"]["p"]
        d = model_info["parameters"]["d"]
        q = model_info["parameters"]["q"]

    # Если авто-подбор параметров
    if auto_select:
        logger.info("Автоматический подбор параметров ARIMA...")
        best_aic = float('inf')
        best_order = None
        for p_test in range(1, 3):
            for q_test in range(1, 3):
                try:
                    model_test = ARIMA(series_for_fit, order=(p_test, 1, q_test))
                    res = model_test.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p_test, 1, q_test)
                except Exception as e:
                    logger.debug(f"Ошибка при подборе параметров p={p_test}, q={q_test}: {e}")

        if best_order is None:
            best_order = (1, 1, 1)
            logger.warning(f"Не удалось найти оптимальные параметры. Используем {best_order}.")
        else:
            logger.info(f"Подобранные параметры: p={best_order[0]}, d={best_order[1]}, q={best_order[2]} (AIC={best_aic:.2f})")

        p, d, q = best_order
    else:
        # Если auto_select=False, параметры p,d,q должны быть заданы
        if p is None or d is None or q is None:
            logger.error("Если auto_select=False, параметры p, d, q должны быть заданы.")
            raise ValueError("Если auto_select=False, параметры p, d, q должны быть заданы.")
        logger.info(f"Используем заданные параметры ARIMA: p={p}, d={d}, q={q}")

    # Обучение модели
    logger.info(f"Обучение модели ARIMA для {ticker} (p={p}, d={d}, q={q})...")
    model = ARIMA(series_for_fit, order=(p, d, q),
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit()

    # Генерация или переиспользование ID
    model_id = model_id if (overwrite and model_id) else f"model_{ticker}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Модель ARIMA для {ticker} успешно обучена. ID модели: {model_id}")


    # Прогноз
    preds = []
    if forecast_steps > 0:
        logger.info(f"Выполнение прогноза на {forecast_steps} шагов...")
        forecast = model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)

        preds = [
            (forecast_mean.index[i],    # Дата предсказания
             forecast_mean.iloc[i],     # Значение предсказания
             conf_int.iloc[i, 0],       # Нижняя граница доверительного интервала
             conf_int.iloc[i, 1])       # Верхняя граница доверительного интервала
            for i in range(forecast_steps)
        ]
        logger.info(f"Прогноз выполнен на {forecast_steps} шагов.")
        logger.debug(f"Предсказания: {preds}")

    # Расчёт метрик, если forecast_steps > 0
    metrics = None
    if forecast_steps > 0:
        y_true = data['Price'].iloc[-forecast_steps:].values
        y_pred = [p[1] for p in preds]
        metrics = compute_metrics(y_true, np.array(y_pred))
        logger.info(f"Метрики для {ticker}: {metrics}")

    # Формируем единый словарь для сохранения
    model_info = {
        "ticker": ticker,
        "id": model_id,
        "model": model,
        "model_name": model_name,
        "data": data,   # Если нужно хранить исходные данные
        "parameters": {
            "auto_select": auto_select,
            "p": p,
            "d": d,
            "q": q,
            "steps": forecast_steps,
        }
    }
    if metrics:
        model_info["metrics"] = metrics

    # Добавление предсказаний в модель
    if preds:
        model_info["forecast"] = {
            "steps": forecast_steps,
            "predictions": preds,
            "last_date": data.index[-1] if not data.empty else None,
        }

    # ЕДИНОКРАТНОЕ сохранение модели (уже с метриками, если есть)
    save_model(save_to, ticker, model_id, model_name, model_info)

    return (preds,model_id,model_name)
