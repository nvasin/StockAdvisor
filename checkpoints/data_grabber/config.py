# Параметры для Yahoo Finance
yahoo_tickers = ['AAPL', 'MSFT', 'GOOGL']
yahoo_start_date = '2024-01-01'
yahoo_end_date = '2024-11-02'
yahoo_interval = '15m'  # Допустимые интервалы: '1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d', '5d', '1wk', '1mo', '3mo'

# Параметры для MOEX
moex_tickers = ['SBER', 'GAZP', 'LKOH']
#moex_tickers = ['LKOH', 'SBER', 'GAZP', 'TATN', 'YDEX', 'GMKN', 'TCSG','NVTK','SNGS','PLZL','SNGSP','SBERP','CHMF','IRAO','NLMK','OZON','RUAL','TATNP','MOEX','FIVE','MGNT','ALRS','MTSS','RTKM','AFLT','AGRO','VTBR','PHOR','POSI','TRNFP','CBOM','AFKS','VKCO','FLOT','BSPB','FEES','ENPG','ASTR','MSNG','SELG','UPRO','MTLR','HYDR','LEAS','SMLT','MTLRP']
moex_start_date = '2024-10-01'
moex_end_date = '2024-11-01'
moex_interval = '10m'  # Допустимые интервалы: '1m', '10m', '1h', '1d'

data_directories = {
    'yahoo': 'data/yahoo',
    'moex': 'data/moex'
}