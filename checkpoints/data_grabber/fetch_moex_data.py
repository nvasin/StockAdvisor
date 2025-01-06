import requests
import pandas as pd
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, clear_output

def parse_interval(interval_str):
    mapping = {
        '1m': 1,
        '10m': 10,
        '1h': 60,
        '1d': 24
    }
    if interval_str in mapping:
        return mapping[interval_str]
    else:
        raise ValueError(f"Недопустимый интервал '{interval_str}' для MOEX.")

def fetch_ticker_data(ticker, start_date, end_date, interval, interval_value, data_dir, worker_id, status_updates):
    all_data = []
    start_index = 0

    while True:
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
        params = {
            'from': start_date,
            'till': end_date,
            'interval': interval_value,
            'start': start_index
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data_json = response.json()
            candles_data = data_json.get('candles')
            if not candles_data or not candles_data['data']:
                status_updates[worker_id] = f"worker {worker_id}: загрузка {ticker} [Нет данных на странице {start_index}]"
                break

            columns = candles_data['columns']
            data = candles_data['data']
            df = pd.DataFrame(data, columns=columns)
            all_data.append(df)
            records_retrieved = len(df)

            status_updates[worker_id] = f"worker {worker_id}: загрузка {ticker} [{records_retrieved} записей из {start_index}]"
            display_status(status_updates)

            if records_retrieved < 100:
                break
            else:
                start_index += 100 
            time.sleep(0.1)  # Задержка между запросами
        else:
            status_updates[worker_id] = f"worker {worker_id}: загрузка {ticker} [Ошибка {response.status_code}]"
            display_status(status_updates)
            break

    if all_data:
        data = pd.concat(all_data, ignore_index=True)
        data = data.rename(columns={
            'begin': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'value': 'Value',
            'volume': 'Volume'
        })
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        data['Open'] = data['Open'].astype(float)
        data['High'] = data['High'].astype(float)
        data['Low'] = data['Low'].astype(float)
        data['Close'] = data['Close'].astype(float)
        data['Volume'] = data['Volume'].astype(float)
        data['Ticker'] = ticker
        data = data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        filename = os.path.join(data_dir, f"{ticker}_moex_{interval}_data.csv")
        data.to_csv(filename, index=False)
        status_updates[worker_id] = f"worker {worker_id}: загрузка {ticker} [Завершено]"
        display_status(status_updates)
    else:
        status_updates[worker_id] = f"worker {worker_id}: загрузка {ticker} [Нет данных]"
        display_status(status_updates)

def display_status(status_updates):
    clear_output(wait=True)
    for worker_id, status in status_updates.items():
        print(status)

def fetch_moex_data(tickers, start_date, end_date, interval, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    interval_value = parse_interval(interval)

    status_updates = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(fetch_ticker_data, ticker, start_date, end_date, interval, interval_value, data_dir, i + 1, status_updates)
            for i, ticker in enumerate(tickers)
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Произошла ошибка: {e}")