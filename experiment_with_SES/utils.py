import os
import numpy as np
import math
import scipy as sc
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
pd.options.plotting.backend = "plotly"

# Simple Exponential Smoothing
# x <array Tx1>- time series,
# h <scalar> - forecasting delay
# Params <dict> - dictionary with
#    alpha <scalar in [0,1]> - smoothing parameter

def SimpleExponentialSmoothing(x, h=1, params={'alpha':0.1}, freq = None):
    T = len(x)
    alpha = params['alpha']

    # retrieve date frequency in ts
    if freq is None:
      freq = pd.infer_freq(x.index)


    # prepare Forecast pd.Series to put forecasted values in it
    forecast = pd.Series(index = x.index.append(pd.date_range(x.index[-1], periods=h+1, freq=freq)[1:]), name = 'fcst '+x.name)

    # some checks for alpha parameter
    if alpha>1:
        w.warn('Alpha can not be more than 1')
        #alpha = 1
        return forecast
    if alpha<0:
        w.warn('Alpha can not be less than 0')
        #alpha = 0
        return forecast

    # initialization
    y = x.iloc[0]

    # forecast all ts step-by-step
    for cntr in range(T):
        if not math.isnan(x.iloc[cntr]):
            if math.isnan(y):
                y=x.iloc[cntr]
            y = alpha*x.iloc[cntr] + (1-alpha)*y  # = y + alpha*(x[cntr]-y)
        #else do not nothing
        forecast.iloc[cntr+h] = y # academic forecast, for ML training
        # forecast_production = forecast
        # forecast_production.iloc[T:] = forecast_production.iloc[T+h] # production forecast

    return forecast #, forecst.iloc[T+h]


def build_forecast(h, ts, alg_name, alg_title, params, step='D'):
  'grid'

  FRC_TS = dict()

  for p in params:
      frc_ts = pd.DataFrame(columns = ts.columns)

      for cntr in ts.columns:
          frc_ts[cntr] = eval(alg_name)(ts[cntr], h, p)

#         frc_ts.columns = frc_ts.columns+('%s %s' % (alg_title, p))
      FRC_TS['%s %s' % (alg_title, p)] = frc_ts

  return FRC_TS


def plot_ts_forecast(ts, frc_ts, ts_num=0, alg_title=''):
    frc_ts.columns = ts.columns+'; '+alg_title
    ts[[ts.columns[ts_num]]].merge(frc_ts[[frc_ts.columns[ts_num]]], how = 'outer', left_index = True, right_index = True)\
      .plot().update_layout(height=350, width=1300,
                  xaxis_title="time ticks",
                  yaxis_title="ts and forecast values").show()
    return


# intoduce loss function
def qualityMAPE(x,y):
    # Mean absolute percentage error
    # x,y - pandas structures
    # x - real values
    # y - forecasts
  qlt = ((x-y).abs()/x).replace([np.inf, -np.inf], np.nan)
  return qlt.mean() , (x-y).abs()    