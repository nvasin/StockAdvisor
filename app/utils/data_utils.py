from utils.logger import logger

import pandas as pd
import os
import glob
from statsmodels.tsa.stattools import adfuller
from typing import List, Tuple, Union


# TODO: вынести переменные в конфиг, а TEST_DAYS в параметры функции
TEST_DAYS = 7  # Число дней для тестовой выборки

def collect_files(file_dir: str, pattern: str) -> List[str]:
    """
    Получает список файлов, соответствующих заданной директории и шаблону.

    Аргументы:
    - file_dir: str - Директория для поиска файлов.
    - pattern: str - Шаблон поиска (например, "*.csv").

    Возвращает:
    - List[str]: Список путей к найденным файлам. Если файлов нет, возвращает пустой список.
    """
    
    logger.debug(f"Ищем файлы в директории: {file_dir} с шаблоном: {pattern}")
 
    file_path = glob.glob(os.path.join(file_dir, pattern))
    
    if file_path:
        for fp in file_path:
            logger.info(f"Обнаружен файл: {fp}")
        return file_path
    else:
        logger.warning("Не найдено ни одного CSV-файла в директории данных!")
        return []

def process_time_series(fp: str) -> Tuple[bool, Union[pd.DataFrame, None]]:
    """
    Обрабатывает данные временного ряда из CSV-файла.

    1. Проверяет наличие всех необходимых столбцов.
    2. Удаляет строки с пропущенными или некорректными данными.
    3. Агрегирует данные с поминутного уровня до дневного (среднее значение).
    4. Восстанавливает ежедневную частоту данных и заполняет пропуски интерполяцией.

    Аргументы:
    - fp (str): Путь к CSV-файлу.

    Возвращает:
    - Tuple[bool, Union[pd.DataFrame, None]]:
        (True, DataFrame): Если данные успешно обработаны.
        (False, None): Если произошла ошибка.
    """

    required_columns = [
        "Datetime", "Open", "High", "Low", "Close", "Volume", "Ticker", "IsMissing",
        "Rolling_Mean", "Rolling_Std", "Rolling_Median", "Upper_Bound", "Lower_Bound",
        "IsOutlier", "Close_Replaced", "Close_Smoothed", "Final_Close"
    ]

    try:
        df = pd.read_csv(fp, parse_dates=['Datetime'])
        logger.info(f"Файл {os.path.basename(fp)} взят в обработку.")
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {os.path.basename(fp)}: {e}")
        return False, None

    # Проверка наличия всех необходимых столбцов
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Файл {os.path.basename(fp)} пропущен. Отсутствуют столбцы: {', '.join(missing_columns)}")
        return False, None

    # Фильтрация данных
    logger.debug(f"Фильтрация данных: было записей {len(df)}")

    df = df.sort_values('Datetime')
    df = df[df['IsMissing'] == 0].dropna(subset=['Final_Close'])

    logger.debug(f"Фильтрация данных: стало записей {len(df)}")

    if df.empty:
        logger.error(f"Файл {os.path.basename(fp)} содержит только пропуски или некорректные данные.")
        return False, None

    logger.debug(f"Агрегация по дням: было записей {len(df)}")
    df['Date'] = pd.to_datetime(df['Datetime'].dt.date)
    daily = (
        df.groupby('Date')
          .agg({'Final_Close': 'mean'})
          .reset_index()
          .rename(columns={'Final_Close': 'Price'})
    )
    logger.debug(f"Агрегация по дням: стало записей {len(daily)}")
    
    # Восстановление частоты и заполнение пропусков
    daily = daily.set_index('Date').asfreq('D')
    daily['Price'] = daily['Price'].interpolate(method='time')
    logger.info(f"Файл {os.path.basename(fp)} обработан успешно: {len(daily)} строк.")

    return True, daily

def split_train_test(daily: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит данные на обучающую (train) и тестовую (test) выборки.

    Аргументы:
    - daily: pd.DataFrame - Данные временного ряда.
    - test_days: int - Количество последних дней для тестовой выборки.

    Возвращает:
    - Tuple[pd.DataFrame, pd.DataFrame]:
        - train: pd.DataFrame - Обучающая выборка.
        - test: pd.DataFrame - Тестовая выборка.
    """
    logger.debug(f"Разделение данных: всего {len(daily)} дней, тестовая выборка {test_days} дней.")
    
    if len(daily) <= test_days:
        logger.warning("Данных меньше или равно количеству тестовых дней. Обучающая выборка будет пустой.")
    
    train = daily.iloc[:-test_days].copy()
    test = daily.iloc[-test_days:].copy()

    logger.info(f"Данные разделены: train - {len(train)} дней, test - {len(test)} дней.")

    return train, test


def stationarize(train):
    """
    ->check_stationarity
    Проверка на стационарность.
        Если p-value > 0.05, делаем разность первого порядка.
    Возвращаем (DataFrame, порядок d).
    """
    adf_res = adfuller(train['Price'].dropna())
    if adf_res[1] > 0.05:
        train['Diff'] = train['Price'].diff().dropna()
        st_train = train.dropna(subset=['Diff'])
        return st_train, 1
    return train, 0

def check_stationarity(train: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Проверяет временной ряд на стационарность с помощью теста ADF.

    Если p-value > 0.05, применяется разность первого порядка (d=1),
    чтобы сделать ряд стационарным. Если ряд уже стационарный, возвращается исходный ряд.

    Аргументы:
    - train (pd.DataFrame): Входной DataFrame с колонкой 'Price'.

    Возвращает:
    - Tuple[pd.DataFrame, int]:
        - DataFrame: Преобразованный DataFrame, где добавлена колонка 'Diff', если ряд нестационарный.
        - int: Порядок разности (0, если стационарный, 1, если применена разность первого порядка).
    """
    try:
        logger.info("Начинается проверка на стационарность.")
        adf_res = adfuller(train['Price'].dropna())
        p_value = adf_res[1]
        logger.info(f"ADF Statistic: {adf_res[0]:.4f}, p-value: {p_value:.4f}")

        if p_value > 0.05:
            logger.info("Ряд нестационарный. Применяем разность первого порядка (d=1).")
            train['Diff'] = train['Price'].diff()
            st_train = train.dropna(subset=['Diff'])
            logger.info(f"Ряд преобразован. Количество строк после разности: {len(st_train)}.")
            return st_train, 1

        logger.info("Ряд стационарный. Преобразование не требуется (d=0).")
        return train, 0

    except Exception as e:
        logger.error(f"Ошибка при проверке на стационарность: {e}")
        raise