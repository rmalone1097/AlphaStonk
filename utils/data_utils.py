import csv
import math
import time
import datetime
import os
import requests
import pandas as pd
import pytz
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import finnhub

from pathlib import Path
from datetime import timedelta, date, datetime
from scipy.interpolate import interp1d
from typing import cast
from urllib3 import HTTPResponse
from polygon import RESTClient
from tqdm import tqdm

POLYGON_API_KEY = 'jGYQQMOIgDQ9c3uefzYyi2AJLcqLXZzM'
FINNHUB_API_KEY = 'cfklqe9r01qokcgl4g20cfklqe9r01qokcgl4g2g'
dirname = os.path.dirname(__file__)

def finnhub_data_writer(tickers, start_stamp, end_stamp=int(time.time()), timespan=1, dir=Path.home() / 'data'):
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    original_start_stamp = start_stamp

    for ticker in tickers:
        # Go two weeks at a time to get all data
        start_datetime = datetime.fromtimestamp(original_start_stamp)
        span_datetime = start_datetime + timedelta(days=14)
        span_stamp = int(datetime.timestamp(span_datetime))
        print(finnhub_client.stock_candles(ticker, str(timespan), original_start_stamp, span_stamp))
        df = pd.DataFrame(finnhub_client.stock_candles(ticker, str(timespan), original_start_stamp, span_stamp))

        while span_stamp < end_stamp:
            start_stamp = span_stamp
            start_datetime = datetime.fromtimestamp(start_stamp)
            span_datetime = start_datetime + timedelta(days=14)
            span_stamp = int(datetime.timestamp(span_datetime))
            new_df = pd.DataFrame(finnhub_client.stock_candles(ticker, str(timespan), start_stamp, span_stamp))
            df = pd.concat([df, new_df])
            print(span_datetime)

        df.to_pickle(dir / str(ticker + '_' + str(original_start_stamp) + '_' + str(end_stamp) + '_' + str(timespan) + '_raw.pkl'))
    
    return df

# Date format YYYY-MM-DD
#TODO: build in support to handle NaN's (Thanksgiving)
def write_data_to_new_file(ticker:str, multiplier:int, start_date:str, end_date:str, dir:str=dirname+'/', timespan:str='minute'):
    client = RESTClient(POLYGON_API_KEY)
    aggs = cast(
        HTTPResponse,
        client.get_aggs(
            ticker,
            multiplier,
            timespan,
            start_date,
            end_date,
            raw=False,
        ),
    )

    csv_header = ['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transacitons']
    path = dir + ticker + '_' + timespan + '_' + start_date + '.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(csv_header)
        for agg in tqdm(aggs):
            time = str(datetime.datetime.fromtimestamp(int(agg.timestamp / 1000), pytz.timezone('US/Eastern')))[0:19]
            row = [time, agg.timestamp, agg.open, agg.high, agg.low, agg.close, agg.volume, agg.vwap, agg.transactions]
            if str(agg.transactions).isnumeric():
                writer.writerow(row)
    return path

def append_data_to_file(ticker:str, multiplier:int, start_date:str, end_date:str, path:str, timespan:str='minute'):
    client = RESTClient(POLYGON_API_KEY)
    aggs = cast(
        HTTPResponse,
        client.get_aggs(
            ticker,
            multiplier,
            timespan,
            start_date,
            end_date,
            raw=False,
        ),
    )

    #csv_header = ['time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transacitons']
    #path = dir + ticker + '_' + timespan + '_' + start_date + '.csv'
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #writer.writerow(csv_header)
        for agg in tqdm(aggs):
            time = str(datetime.datetime.fromtimestamp(int(agg.timestamp / 1000), pytz.timezone('US/Eastern')))[0:19]
            row = [time, agg.timestamp, agg.open, agg.high, agg.low, agg.close, agg.volume, agg.vwap, agg.transactions]
            writer.writerow(row)
    return path

def df_builder(ticker:str, pickle_dir):
    raw_df = pd.read_pickle(pickle_dir)
    array = raw_df.to_numpy()
    prev_i = 0
    repeat_max = 30

    # Splitting into subarrays and concatenating at the end to try to speed up computation time
    sub_arrays = 20
    resolution = int(array.shape[0] / sub_arrays)
    built_array_list = []
    sub_array_counter = 0
    #m_counter = []

    while sub_array_counter < sub_arrays:
        sub_array = array[resolution*sub_array_counter:resolution*(sub_array_counter + 1), :]
        built_array = np.array([sub_array[0, :]])
        prev_i = 0

        for i in tqdm(range(1, sub_array.shape[0]-1)):
            prev_dt = datetime.fromtimestamp(sub_array[i-1, 5])
            dt = datetime.fromtimestamp(sub_array[i, 5])
            delta = dt - prev_dt
            m_delta = int(delta.total_seconds() / 60)
            row = np.array([sub_array[i-1, :]])
            #m_counter.append(m_delta)

            if m_delta > 1 and dt.hour >= 7 and dt.hour <= 14:
                built_array = np.concatenate((built_array, sub_array[prev_i+1 : i]))
                built_array = np.concatenate((built_array, np.repeat(row, min(m_delta, repeat_max), axis=0)))
                #m_counter = m_counter + [m_delta]*min(m_delta*repeat_max)
                prev_i = i
        sub_array_counter += 1

        built_array_list.append(built_array)
    
    complete_array = np.vstack(built_array_list)
    df = pd.DataFrame(complete_array, columns=['close', 'high', 'low', 'open', 'status', 'timestamp', 'volume'])

    daily_candle_counter = []
    prev_counter = 0
    prev_date = 0
    spaced_entries = dict()
    for i, row in tqdm(enumerate(df.itertuples(index=False)), total=len(df)):
        counter = 0
        date = datetime.fromtimestamp(int(row.timestamp), pytz.timezone('US/Eastern'))
        # Make daily candle counter during trading hours
        if prev_counter != 0:
            if date.hour == 16 and date.minute == 0:
                counter = 0
            else:
                counter = prev_counter + 1
        elif date.hour == 9 and date.minute == 30:
            counter = 1
        daily_candle_counter.append(counter)
        prev_counter = counter

        '''# Look for missing data points and repeat previous data points to make up for it
        # TODO: Read CSV into lists to improve efficiency of appending
        if prev_date != 0 and (prev_date + timedelta(minutes=1)).minute != date.minute:
            print('Missing entry', date)
            #df.loc[i - 0.5] = df.loc[i-1]
            total_seconds = (date - prev_date).total_seconds()
            spaced_entries[i] = int(total_seconds / 60)
            #daily_candle_counter.append(0)
            #counter += 1
        prev_date = date
        #prev_row = df.iloc[i]
    #print(len(daily_candle_counter))'''
    
    '''d = df.to_dict(orient='list')
    idx_displacement = 0
    for key, value in tqdm(spaced_entries.items()):
        while value != 1:
            idx = key + idx_displacement - 1
            for list in d.values():
                list.insert(key + idx_displacement, list[idx])
            daily_candle_counter.insert(idx, 0)
            idx_displacement += 1
            value -= 1'''
    
    #print(len(daily_candle_counter))
    #new_df = pd.DataFrame.from_dict(d)
    df['daily_candle_counter'] = daily_candle_counter
    df = df.fillna(0)
    df.to_pickle(Path.home() / 'data' / 'AAPL_built.pkl')

    return raw_df, df

# Polygon outputs 5000 data points max about 5 days with pre/post data
def fetch_all_data(ticker:str, multiplier:int, start_date:str, end_date:str, dir=dirname+'/', timestamp:str='minute'):
    firstfetch = True
    counter = 0
    start_date_obj = date(year=int(start_date[0:4]), month=int(start_date[5:7]), day=int(start_date[8:10]))
    end_date_obj   = date(year=int(end_date[0:4]), month=int(end_date[5:7]),  day=int(end_date[8:10]))

    current_date_obj = start_date_obj

    while current_date_obj < end_date_obj:
        datedelta = (end_date_obj - start_date_obj).days
        if datedelta > 5:
            datedelta = 5
        
        if firstfetch:
            path = write_data_to_new_file(ticker, multiplier, str(current_date_obj), str(current_date_obj + timedelta(days=datedelta)), dir)
        else:
            append_data_to_file(ticker, multiplier, str(current_date_obj), str(current_date_obj + timedelta(days=datedelta)), path)

        current_date_obj += timedelta(days=datedelta+1)
        firstfetch = False
        counter += 1 

        if counter == 5:
            time.sleep(60)
            counter = 0
            str(current_date_obj + timedelta(days=datedelta))

def add_indicators(df):
    df.set_index(pd.DatetimeIndex(df["time"]), inplace=True)
    trading_df = df.loc[df['daily_candle_counter'] > 0]

    # Minute length EMA
    trading_df['ema_5'] = ta.ema(trading_df['close'], length=5)
    trading_df['ema_10'] = ta.ema(trading_df['close'], length=10)
    trading_df['ema_15'] = ta.ema(trading_df['close'], length=15)
    trading_df['ema_25'] = ta.ema(trading_df['close'], length=25)
    trading_df['ema_40'] = ta.ema(trading_df['close'], length=40)
    trading_df['ema_65'] = ta.ema(trading_df['close'], length=65)
    trading_df['ema_170'] = ta.ema(trading_df['close'], length=170)
    trading_df['ema_250'] = ta.ema(trading_df['close'], length=250)
    trading_df['ema_360'] = ta.ema(trading_df['close'], length=360)
    trading_df['ema_445'] = ta.ema(trading_df['close'], length=445) 
    trading_df['ema_900'] = ta.ema(trading_df['close'], length=900)
    trading_df['ema_1000'] = ta.ema(trading_df['close'], length=1000)

    #day length ema (5, 10, 20, 50, 100)
    trading_df['ema_5_day'] = ta.ema(trading_df['close'], length=5*390)
    trading_df['ema_10_day'] = ta.ema(trading_df['close'], length=10*390)
    trading_df['ema_20_day'] = ta.ema(trading_df['close'], length=20*390)
    trading_df['ema_50_day'] = ta.ema(trading_df['close'], length=50*390)
    trading_df['ema_100_day'] = ta.ema(trading_df['close'], length=100*390)

    return trading_df

def plot_df_slice(df, starting_index=0, ending_index=30):
    taplots = []
    taplots += [mpf.make_addplot(df['ema_5'].iloc[starting_index:ending_index], color='red', panel=0),
        mpf.make_addplot(df['ema_10'].iloc[starting_index:ending_index], color='green', panel=0),
        mpf.make_addplot(df['ema_15'].iloc[starting_index:ending_index], color='orange', panel=0),
        mpf.make_addplot(df['ema_25'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_40'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_65'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_170'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_250'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_360'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_445'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_900'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_1000'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_5_day'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_10_day'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_20_day'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_50_day'].iloc[starting_index:ending_index], panel=0),
        mpf.make_addplot(df['ema_100_day'].iloc[starting_index:ending_index], panel=0)]
    mpf.plot(df.iloc[starting_index:ending_index], type='candle', addplot=taplots)

def plot_energy_cloud(df, starting_index=0, ending_index=30):
    taplots = []
    long_rewards = [0]
    short_rewards = [0]
    zero_rewards = [0]
    energies = [0]
    df_slice = df.iloc[starting_index:ending_index]

    for i, row in tqdm(enumerate(df_slice.itertuples(index=False)), total=len(df)):
        if i != 0:
            energy = (row.ema_25 - row.ema_170) / row.ema_170 * 100
            if row.daily_candle_counter > 120 or row.daily_candle_counter == 1:
                reward = energy + ((row.close - row.ema_25) / row.ema_25 * 250)
            else:
                reward = (row.close - row.ema_25) / row.ema_25 * 250

            energies.append(energy)
            long_rewards.append(reward)
            short_rewards.append(-reward)

            if abs(reward) <= 0.4 or row.daily_candle_counter < 15:
                zero_rewards.append(0)
            else:
                zero_rewards.append(-abs(reward))
    
    print('Sum of long rewards: ', sum(long_rewards))
    print('Sum of short rewards: ', sum(short_rewards))
    print('Sum of zero rewards: ', sum(zero_rewards))

    df_slice['long_reward'] = long_rewards
    df_slice['short_reward'] = short_rewards
    df_slice['zero_reward'] = zero_rewards
    df_slice['energy'] = energies

    taplots += [
        mpf.make_addplot(df_slice['ema_25'], panel=0),
        mpf.make_addplot(df_slice['ema_170'], panel=0),
        mpf.make_addplot(df_slice['long_reward'], panel=1, color='green', ylabel='Rewards'),
        mpf.make_addplot(df_slice['short_reward'], panel=1, color='red'),
        mpf.make_addplot(df_slice['zero_reward'], panel=2, ylabel='Zero Reward'),
        mpf.make_addplot(df_slice['energy'], panel=3, color='orange', ylabel='Energy')]
    mpf.plot(df.iloc[starting_index:ending_index], type='candle', addplot=taplots)