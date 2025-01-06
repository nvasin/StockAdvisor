from utils.logger import logger
from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import math
from utils.data_utils import collect_files, process_time_series, split_train_test
from utils.model_storage import (
    TRAINED_ARIMA_MODELS, save_model, delete_model, get_all_models, 
    get_models_by_group, get_models_by_type, get_models_by_ticker, get_model_by_id
)
from models.arima_model import arima_model

# Константы
# TODO вынести
FILE_DIR = '/StockAdvisor/datasets/cleaned'
PATTERN = '*_cleaned.csv'
TEST_DAYS = 7  # число дней для тестовой выборки

# Инициализация приложения FastAPI
app = FastAPI()



@app.on_event("startup")
def load_and_train_models():
    """
    При запуске приложения:
    1) Считываем CSV для каждого тикера.
    2) Делим на train/test (последние 7 дней — тест).
    3) Считаем прогноз на 3 дня (для метрик).
    4) Сохраняем только "overall" метрики в models_storage.
    """
    logger.info("Запуск приложения и загрузка моделей.")
    file_paths = collect_files(FILE_DIR, PATTERN)

    if not file_paths:
        logger.error("Приложение запущено без загрузки данных. Файлы отсутствуют. Возможна некорректная работа.")
        return

    for fp in file_paths:
        base_name = os.path.basename(fp)
        ticker = base_name.split("_cleaned")[0]
        logger.info(f"Обработка файла {fp} для тикера {ticker}.")

        success, result = process_time_series(fp)

        if not success:
            logger.error(f"Ошибка обработки файла {base_name}")
            continue

        daily = result  # Это df, если success == True
        if len(daily) < 14:
            logger.warning(f"Недостаточно данных в {fp}. Пропускаем.")
            continue

        train, test = split_train_test(daily, TEST_DAYS)

        arima_initial_preds = arima_model(data=train,ticker=ticker,auto_select=True,forecast_steps=3,save_to='auto_initialized',overwrite = True)
        #TODO адаптировать ses_forecast
        #ses_preds = ses_forecast(train.copy(), 3)
        #TODO адаптировать linear_forecast
        #lin_preds = linear_forecast(train.copy(), 3)

        #На текщий момент не реализованы, так как являются слабоперспективными для проекта

    logger.info(f"Модели успешно загружены: {len(TRAINED_ARIMA_MODELS)} групп.")

@app.get("/models")
def list_models():
    """
    Возвращает список всех моделей в JSON-совместимом формате.
    """
    models = get_all_models()

    # Обработка недопустимых значений
    def sanitize_values(data):
        if isinstance(data, dict):
            return {key: sanitize_values(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [sanitize_values(item) for item in data]
        elif isinstance(data, (float, np.float32, np.float64)):
            if math.isnan(data) or math.isinf(data):
                return None
        return data

    sanitized_models = sanitize_values(models)
    logger.info("Список моделей успешно получен.")
    return JSONResponse(content=sanitized_models)


class UserDataPoint(BaseModel):
    Date: str
    Price: float

class FitRequest(BaseModel):
    ticker: str
    forecast_steps: int
    model_type: Optional[str] = None
    auto_select: bool = True
    p: Optional[int] = None
    d: Optional[int] = None
    q: Optional[int] = None
    user_data: Optional[List[UserDataPoint]] = None
    model_id: Optional[str] = None

@app.post("/fit")
def fit_model(request: FitRequest):
    """
    Обучает модель временного ряда, используя системные и/или пользовательские данные.
    """
    logger.info(f"Начало обучения модели для тикера {request.ticker}.")
    ticker = request.ticker
    forecast_steps = request.forecast_steps
    model_type = request.model_type  # можно пока игнорировать
    auto_select = request.auto_select
    p = request.p
    d = request.d
    q = request.q
    user_data = request.user_data
    model_id = request.model_id  # Получаем ID модели, если он указан

    if not auto_select:
        # Если auto_select=False, то p, d, q должны быть заданы
        if p is None or d is None or q is None:
            logger.error("При auto_select=false необходимо указать параметры p, d, q.")
            raise HTTPException(
                status_code=400,
                detail="При auto_select=false параметры p, d, q должны быть указаны."
            )

    df_system = pd.DataFrame()
    df_user = pd.DataFrame()

    file_path = os.path.join(FILE_DIR, f"{ticker}_cleaned.csv")
    
    if os.path.exists(file_path):
        success, daily = process_time_series(file_path)
        if not success or daily is None:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка обработки данных для {ticker}."
            )
        df_system = daily
    else:
        logger.warning(f"Системные данные для {ticker} не найдены.")

    if user_data:
        df_user = pd.DataFrame([{"Date": x.Date, "Price": x.Price} for x in user_data])
        df_user['Date'] = pd.to_datetime(df_user['Date'])
        df_user = df_user.set_index('Date').sort_index()

 
    if not df_system.empty and not df_user.empty:
        df_merged = pd.concat([df_system, df_user], axis=0)
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
        df_merged = df_merged.sort_index()
        data_for_train = df_merged
    else:
        data_for_train = df_user if not df_user.empty else df_system

    if data_for_train.empty:
        logger.error("Обучение невозможно: итоговый набор данных пуст.")
        raise HTTPException(
            status_code=400,
           detail="Итоговый набор данных пуст."
        )

    train = data_for_train

    preds, model_id, model_name = arima_model(
        data=train.copy(),
        ticker=ticker,
        save_to="fit_endpoint",  
        auto_select=auto_select,
        p=p,
        d=d,
        q=q,
        forecast_steps=forecast_steps,
        overwrite=False
    )

    preds_serializable = [
        tuple(str(item) for item in pred)
        for pred in preds 
    ]
    
    response = {
        "ticker": ticker,
        "forecast_steps": forecast_steps,
        "auto_select": auto_select,
        "model_id": model_id, 
        "model_name": model_name, 
        "data_count": len(train),
        "predictions": preds_serializable
    }
    logger.info(f"Модель для {ticker} успешно обучена. ID модели: {model_id}.")
    return JSONResponse(content=response)



@app.get("/predict")
def get_predictions(model_id: List[str] = Query(...)):
    """
    Получает прогнозы для указанных моделей.
    """
    logger.info("Получение прогнозов по ID моделей.")
    results = []

    for model_id in model_id:
        # Fetch the model using its ID
        model_entry = get_model_by_id(model_id)
        if not model_entry:
            logger.warning(f"Модель с ID {model_id} не найдена.")
            results.append({
                "model_id": model_id,
                "error": "Модель не найдена."
            })
            continue

        model_info = model_entry["info"]
        predictions = model_info.get("forecast", {}).get("predictions", [])

        if not predictions:
            results.append({
                "model_id": model_id,
                "error": "No predictions found for this model"
            })
            continue

        results.append({
            "model_id": model_id,
            "ticker": model_info["ticker"],
            "forecast_steps": model_info["parameters"]["steps"],
            "predictions": predictions,
        })

    if not results:
        logger.error("Прогнозы не найдены.")
        raise HTTPException(status_code=404, detail="Прогнозы не найдены.")

    return {"models": results}
