import pandas as pd
import numpy as np
import datetime

from tqdm import tqdm
from data_utils import *

def reset(df):
    window_days = 5
    first_found = False
    first_trading_stamp = 0
    for i, row in enumerate(df.itertuples()):
        if row.daily_candle_counter != 0 and first_found == False:
            first_found = True
            # First market open data point
            first_valid_day = datetime.datetime.fromtimestamp(int(row.timestamp) / 1000, pytz.timezone('US/Eastern'))
            # Name, or set index of df, is expected to be timestep (milliseconds) and will be used to locate
            # data points of interest
            first_valid_name = i
            # Calculation of first trading date, window days + 1
            first_trading_day = first_valid_day + timedelta(days=window_days + 1)
            assert (first_trading_day.hour, first_trading_day.minute) == (9, 30), "Calculation of first trading point is incorrect"
            # Calculation of first trading point on first trading day (9:30AM EST on first trading day)
            first_trading_stamp = int(round(first_trading_day.timestamp() * 1000))
        if row.timestamp == first_trading_stamp:
            first_trading_name = i
            break
    # The state of the environment is the data slice that the agent will have access to to make a decision
    df_slice = df.iloc[first_valid_name:first_trading_name]
    state_idx = [first_valid_name, first_trading_name]
    return df_slice, state_idx

def step(df, state_idx):
    # Fetch first and last index of the window and add 1
    first_idx, last_idx = state_idx[0] + 1, state_idx[1] + 1
    # If data point after last is after market close, find the next market open point
    if df.iloc[last_idx]['daily_candle_counter'] == 0:
        for i, row in enumerate(df.iloc[last_idx:].itertuples()):
            # add i to last_idx and first idx to keep slice length consistent
            if row.daily_candle_counter != 0:
                first_idx, last_idx = first_idx + i, last_idx + i
                break
    df_slice = df.iloc[first_idx:last_idx]
    state_idx = [first_idx, last_idx]
    return df_slice, state_idx

df = df_builder('SPY_minute_2020-07-29.csv')
df_slice_0, state_idx_0 = reset(df)
df_slice_1, state_idx_1 = step(df, [state_idx_0[0]+390, state_idx_0[1]+390])
df_slice_2, state_idx_2 = step(df, state_idx_1)
df_slice_3, state_idx_3 = step(df, state_idx_2)