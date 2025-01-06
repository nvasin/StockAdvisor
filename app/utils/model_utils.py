from utils.logger import logger

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from typing import Dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Вычисляет метрики MAE, MAPE и R^2.

    Аргументы:
    - y_true: np.ndarray - Истинные значения.
    - y_pred: np.ndarray - Предсказанные значения.

    Возвращает:
    - dict: Метрики MAE, MAPE, R^2.
    """
    logger.info("Начало вычисления метрик MAE, MAPE и R^2.")
    try:
        mae = mean_absolute_error(y_true, y_pred)
        logger.debug(f"MAE вычислено: {mae:.4f}")

        mape = mean_absolute_percentage_error(y_true, y_pred)
        logger.debug(f"MAPE вычислено: {mape:.4f}")

        r2 = r2_score(y_true, y_pred)
        logger.debug(f"R^2 вычислено: {r2:.4f}")

        logger.info("Метрики успешно вычислены.")
        return {
            "mae": mae,
            "mape": mape,
            "r2": r2
        }

    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик: {e}")
        raise