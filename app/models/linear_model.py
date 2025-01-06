import pandas as pd
from sklearn.linear_model import LinearRegression

def linear_forecast(train, steps=3):
    """
    1. Добавляем лаги 1 и 2.
    2. Обучаем линейную регрессию.
    3. Прогнозируем пошагово (3 дня).
    Возвращаем список:
      [
        (pred_day1, low_day1, high_day1),
        (pred_day2, low_day2, high_day2),
        (pred_day3, low_day3, high_day3)
      ]
    """
    # Делаем reset_index, чтобы "Date" была просто колонкой
    train = train.reset_index(drop=False).copy()

    train['Lag1'] = train['Price'].shift(1)
    train['Lag2'] = train['Price'].shift(2)
    train.dropna(inplace=True)

    X = train[['Lag1', 'Lag2']]
    y = train['Price']

    model = LinearRegression()
    model.fit(X, y)

    preds = []

    # Последние значения лагов
    last_lag1 = train.iloc[-1]['Price']   # это Price последнего дня
    last_lag2 = train.iloc[-1]['Lag1']    # это сдвинутая на 1 день цена

    for _ in range(steps):
        # Предикт
        input_vec = [[last_lag1, last_lag2]]
        f = model.predict(input_vec)[0]

        lo = f * 0.95
        hi = f * 1.05
        preds.append((f, lo, hi))

        # Обновляем лаги для следующего шага:
        last_lag2 = last_lag1
        last_lag1 = f

    return preds
