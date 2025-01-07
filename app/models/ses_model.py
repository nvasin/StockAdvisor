import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

def ses_forecast(train, steps=3):
    """
    Возвращает список из steps предсказаний:
    [
      (pred_day1, low_day1, high_day1),
      (pred_day2, low_day2, high_day2),
      (pred_day3, low_day3, high_day3)
    ]
    """
    preds = []
    cur = train['Price'].copy()
    idx = cur.index[-1]

    for _ in range(steps):
        model = SimpleExpSmoothing(cur).fit(optimized=True)
        # В логах можно посмотреть alpha
        alpha = model.params['smoothing_level']  
        print(f"[SES] smoothing_level (alpha)={alpha:.4f}")

        f = model.forecast(1).iloc[-1]

        lo = f * 0.95
        hi = f * 1.05
        preds.append((f, lo, hi))

        idx += pd.Timedelta(days=1)
        cur.loc[idx] = f

    return preds
