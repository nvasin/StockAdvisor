from utils.logger import logger
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional


# Глобальный словарь для хранения моделей
TRAINED_ARIMA_MODELS: Dict[str, Dict[str, Dict[str, List[Dict[str, Union[str, dict]]]]]] = {}


def save_model(group: str, ticker: str, model_id: str, model_name: str, model_info: dict) -> None:
    """
    Сохраняет модель в глобальное хранилище.

    Аргументы:
    - group: str - Группа для сохранения модели (например, 'auto_initialized').
    - ticker: str - Название тикера.
    - model_id: str - Уникальный идентификатор модели.
    - model_name: str - Название модели.
    - model_info: dict - Информация о модели.
    """
    # Инициализация группы, если ее нет
    if group not in TRAINED_ARIMA_MODELS:
        TRAINED_ARIMA_MODELS[group] = {}
    if ticker not in TRAINED_ARIMA_MODELS[group]:
        TRAINED_ARIMA_MODELS[group][ticker] = {"versions": []}

    # Список версий для тикера
    versions = TRAINED_ARIMA_MODELS[group][ticker]["versions"]
    version_number = len(versions) + 1
    model_info["version"] = version_number  # Добавляем номер версии
    model_info["saved_at"] = datetime.utcnow().isoformat()  # Добавляем время сохранения

    model_record = {
        "id": model_id,
        "name": model_name,
        "info": model_info,
    }

    versions.append(model_record)

    logger.info(
        f"Модель '{model_name}' для тикера '{ticker}' сохранена в группе '{group}' "
        f"с ID '{model_id}' и номером версии {version_number}."
    )




def delete_model(group: str, ticker: str, model_id: str) -> bool:
    """
    Удаляет модель из хранилища.

    Аргументы:
    - group: str - Группа моделей.
    - ticker: str - Название тикера.
    - model_id: str - ID модели.

    Возвращает:
    - bool: True, если модель удалена, иначе False.
    """
    if group in TRAINED_ARIMA_MODELS and ticker in TRAINED_ARIMA_MODELS[group]:
        versions = TRAINED_ARIMA_MODELS[group][ticker]["versions"]
        for version in versions:
            if version["id"] == model_id:
                versions.remove(version)
                logger.info(f"Модель с ID {model_id} для {ticker} удалена из группы {group}.")
                return True
    logger.warning(f"Не удалось найти модель с ID {model_id} для {ticker} в группе {group}.")
    return False





def get_all_models() -> dict:
    """
    Возвращает список всех моделей во всех группах в JSON-совместимом формате.
    """
    serialized_models = {}

    for group_name, ticker_data in TRAINED_ARIMA_MODELS.items():
        group_result = {}
        for ticker, ticker_dict in ticker_data.items():
            version_list = []
            for version_entry in ticker_dict.get("versions", []):
                info = version_entry.get("info", {})

                forecast_data = info.get("forecast", {})
                predictions_raw = forecast_data.get("predictions", [])
                serialized_predictions = []
                for pred in predictions_raw:
                    if len(pred) == 4:
                        dt, mean_val, lo_val, hi_val = pred
                        if isinstance(dt, (pd.Timestamp, datetime)):
                            dt = dt.isoformat()
                        mean_val = float(mean_val) if isinstance(mean_val, (np.float64, np.float32)) else mean_val
                        lo_val = float(lo_val) if isinstance(lo_val, (np.float64, np.float32)) else lo_val
                        hi_val = float(hi_val) if isinstance(hi_val, (np.float64, np.float32)) else hi_val
                        serialized_predictions.append([dt, mean_val, lo_val, hi_val])
                    else:
                        serialized_predictions.append(pred)

                # Преобразуем last_date (Timestamp -> str)
                last_date = forecast_data.get("last_date")
                if isinstance(last_date, (pd.Timestamp, datetime)):
                    last_date = last_date.isoformat()

                # Формируем удобный словарь для forecast
                forecast_dict = {
                    "steps": forecast_data.get("steps", 0),
                    "predictions": serialized_predictions,
                    "last_date": last_date,
                }

                # Собираем версию в единый словарь
                version_dict = {
                    "id": version_entry.get("id"),
                    "name": version_entry.get("name"),
                    "ticker": info.get("ticker"),  # Например, "ALRS"
                    "parameters": info.get("parameters", {}),
                    "metrics": info.get("metrics", {}),
                    "forecast": forecast_dict,
                    "version": info.get("version"),
                    "saved_at": info.get("saved_at"),
                }

                version_list.append(version_dict)

            group_result[ticker] = {"versions": version_list}
        serialized_models[group_name] = group_result
    logger.info("Сформирован полный список моделей.")
    return serialized_models





def get_models_by_group(group: str) -> Dict[str, Dict[str, List[Dict[str, Union[str, dict]]]]]:
    """
    Возвращает все модели из указанной группы.

    Аргументы:
    - group: str - Группа моделей.

    Возвращает:
    - dict: Словарь моделей.
    """
    logger.info(f"Получение моделей из группы '{group}'.")
    return TRAINED_ARIMA_MODELS.get(group, {})

def get_models_by_type(model_type: str) -> List[dict]:
    """
    Возвращает список всех моделей по типу модели.

    Аргументы:
    - model_type: str - Тип модели (например, 'ARIMA').

    Возвращает:
    - list: Список моделей указанного типа.
    """
    models = []
    for group in TRAINED_ARIMA_MODELS.values():
        for ticker_models in group.values():
            for version in ticker_models["versions"]:
                if version["info"]["model"].__class__.__name__ == model_type:
                    models.append(version)
    logger.info(f"Найдено {len(models)} моделей типа '{model_type}'.")
    return models

def get_models_by_ticker(ticker: str) -> list:
    """
    Возвращает список всех моделей по тикеру.

    Аргументы:
    - ticker: str - Название тикера.

    Возвращает:
    - list: Список моделей для указанного тикера.
    """
    models = []
    for group in TRAINED_ARIMA_MODELS.values():
        if ticker in group:
            models.extend(group[ticker]["versions"])
    logger.info(f"Найдено {len(models)} моделей для тикера '{ticker}'.")
    return models

def get_model_by_id(model_id: str) -> dict:
    """
    Возвращает модель по ID.

    Аргументы:
    - model_id: str - ID модели.

    Возвращает:
    - dict: Модель с указанным ID или None, если не найдена.
    """
    for group in TRAINED_ARIMA_MODELS.values():
        for ticker_models in group.values():
            for version in ticker_models["versions"]:
                if version["id"] == model_id:
                    logger.info(f"Модель с ID '{model_id}' найдена.")
                    return version
    logger.warning(f"Модель с ID '{model_id}' не найдена.")
    return None