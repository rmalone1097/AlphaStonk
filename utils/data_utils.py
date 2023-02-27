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

from functools import reduce
from pathlib import Path
from datetime import timedelta, date
import time
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
        start_datetime = datetime.datetime.fromtimestamp(original_start_stamp)
        span_datetime = start_datetime + timedelta(days=14)
        span_stamp = int(datetime.datetime.timestamp(span_datetime))
        print(finnhub_client.stock_candles(ticker, str(timespan), original_start_stamp, span_stamp))
        df = pd.DataFrame(finnhub_client.stock_candles(ticker, str(timespan), original_start_stamp, span_stamp))

        while span_stamp < end_stamp:
            start_stamp = span_stamp + 60
            start_datetime = datetime.datetime.fromtimestamp(start_stamp)
            span_datetime = start_datetime + timedelta(days=14)
            span_stamp = int(datetime.datetime.timestamp(span_datetime))
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

def df_builder(ticker:str, pickle_dir):
    raw_df = pd.read_pickle(pickle_dir)
    array = raw_df.to_numpy()
    first_found = False
    daily_arrays = []
    daily_array = np.array([])

    i = 1
    while i < array.shape[0] - 2:
        row = np.array([array[i, :]])
        prev_row = np.array([array[i-1, :]])

        # Datetime object made from timestamp of current row
        dt = datetime.datetime.fromtimestamp(array[i, 5], pytz.timezone('US/Eastern'))
        prev_dt = datetime.datetime.fromtimestamp(array[i-1, 5], pytz.timezone('US/Eastern'))
        delta = dt - prev_dt
        m_delta = int(delta.total_seconds() / 60)

        delta_to_end = prev_dt.replace(hour=15, minute=59) - prev_dt
        m_delta_to_end = int(delta_to_end.total_seconds() / 60)

        # Check for first dp in trading day. If no 9:30 dp, starting dp will be dp prior to the first one in trading day
        if first_found == False:
            if dt.time() == datetime.time(9, 30):
                first_found = True
                daily_array = np.array([])
                #i += 1

            elif dt.time() > datetime.time(9, 30) and dt.time() < datetime.time(15, 59):
                delta = dt - dt.replace(hour=9, minute=30)
                m_delta = int(delta.total_seconds() / 60)
                first_found = True
                daily_array = np.repeat(prev_row, m_delta, axis=0)
        
        elif first_found == True:
            # Check for last dp
            if dt.time() == datetime.time(15, 59):
                daily_array = np.concatenate((daily_array, np.repeat(prev_row, m_delta, axis=0)))
                daily_array = np.concatenate((daily_array, row), axis=0)
                daily_arrays.append(daily_array)
                daily_array = np.array([[]])
                first_found = False

            elif dt.time() > datetime.time(15, 59) or m_delta_to_end < m_delta:
                daily_array = np.concatenate((daily_array, np.repeat(prev_row, (m_delta_to_end + 1), axis=0)))
                daily_arrays.append(daily_array)
                daily_array = np.array([[]])
                first_found = False
            
            else:
                if daily_array.size == 0:
                    daily_array = np.repeat(prev_row, m_delta, axis=0)
                else:
                    daily_array = np.concatenate((daily_array, np.repeat(prev_row, m_delta, axis=0)))

        i += 1
        
    complete_array = np.vstack(daily_arrays)
    df = pd.DataFrame(complete_array, columns=[ticker+'_close', ticker+'_high', ticker+'_low', ticker+'_open', 'status', 'timestamp', ticker+'_volume'])

    #df['daily_candle_counter'] = daily_candle_counter
    df['datetime'] = df.apply(lambda row: datetime.datetime.fromtimestamp(row.timestamp), axis=1)
    raw_df['dt'] = raw_df.apply(lambda row: datetime.datetime.fromtimestamp(row.t), axis=1)   
    df = df.fillna(0)

    i_to_remove = []
    black_friday = {1999:26, 2000:24, 2001:23, 2002:29, 2003:28, 2004:26, 2005:25, 2006:24, 2007:23, 2008:28,
                    2009:27, 2010:26, 2011:25, 2012:23, 2013:29, 2014:28, 2015:27, 2016:25, 2017:24, 2018:23,
                    2019:29, 2020:27, 2021:26, 2022:25}
    # Handle holidays

    early_close_candles = 210
    max_candles = 390
    daily_candle_counter = []
    daily_candle = 0
    candle_counter = 0
    candle_counter_log = 0
    for i, row in tqdm(enumerate(df.itertuples(index=True)), total=len(df)):
        dt = row.datetime

        if dt.month == 7: 
            if dt.day == 3:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1

            elif dt.year == 2002 and dt.day == 5:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1
            
            else:
                if max_candles != 390:
                    max_candles = 390
                    candle_counter = 0
                candle_counter += 1

        elif dt.month == 11:
            if dt.day == black_friday[dt.year]:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1
            
            else:
                if max_candles != 390:
                    max_candles = 390
                    candle_counter = 0
                candle_counter += 1

        elif dt.month == 12:
            if dt.day == 24:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1

            elif dt.year == 1999 and dt.day == 31:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1

            elif dt.year == 2003 and dt.day == 26:
                max_candles = early_close_candles
                if candle_counter == 0:
                    delta = dt - dt.replace(hour=7, minute=30)
                    m_delta = int(delta.total_seconds() / 60)
                    candle_counter += m_delta
                candle_counter += 1
            
            else:
                if max_candles != 390:
                    max_candles = 390
                    candle_counter = 0
                candle_counter += 1
            
        else:
            if max_candles != 390:
                max_candles = 390
                candle_counter = 0
            candle_counter += 1
        
        if candle_counter > max_candles and max_candles == early_close_candles:
            i_to_remove.append(i)
        elif candle_counter == candle_counter_log:
            i_to_remove.append(i)
        else:
            daily_candle_counter.append(candle_counter)
        
        candle_counter_log = candle_counter
        
        if candle_counter == max_candles:
            candle_counter = 0
    
    df = df.drop(index=i_to_remove)
    df['daily_candle_counter'] = daily_candle_counter

    df = df.set_index('datetime') 
    file_name = 'df_' + ticker + '_built.pkl'
    df.to_pickle(Path.home() / 'data' / file_name)

    return raw_df, df

def add_indicators(ticker, df):
    trading_df = df

    # Minute length EMA
    trading_df[ticker + '_ema_5'] = ta.ema(trading_df[ticker + '_close'], length=5)
    trading_df[ticker + '_ema_10'] = ta.ema(trading_df[ticker + '_close'], length=10)
    trading_df[ticker + '_ema_15'] = ta.ema(trading_df[ticker + '_close'], length=15)
    trading_df[ticker + '_ema_25'] = ta.ema(trading_df[ticker + '_close'], length=25)
    trading_df[ticker + '_ema_40'] = ta.ema(trading_df[ticker + '_close'], length=40)
    trading_df[ticker + '_ema_65'] = ta.ema(trading_df[ticker + '_close'], length=65)
    trading_df[ticker + '_ema_170'] = ta.ema(trading_df[ticker + '_close'], length=170)
    trading_df[ticker + '_ema_250'] = ta.ema(trading_df[ticker + '_close'], length=250)
    trading_df[ticker + '_ema_360'] = ta.ema(trading_df[ticker + '_close'], length=360)
    trading_df[ticker + '_ema_445'] = ta.ema(trading_df[ticker + '_close'], length=445) 
    #trading_df[ticker + '_ema_900'] = ta.ema(trading_df[ticker + '_close'], length=900)
    #trading_df[ticker + '_ema_1000'] = ta.ema(trading_df[ticker + '_close'], length=1000)

    trading_df[ticker + '_energy'] = (trading_df[ticker + '_ema_25'] - trading_df[ticker + '_ema_170']) / trading_df[ticker + '_ema_170'] * 100
    energy = trading_df.pop(ticker + '_energy')
    df = df.insert(0, ticker + '_energy', energy)

    #day length ema (5, 10, 20, 50, 100)
    #trading_df['ema_5_day'] = ta.ema(trading_df['close'], length=5*390)
    #trading_df['ema_10_day'] = ta.ema(trading_df['close'], length=10*390)
    #trading_df['ema_20_day'] = ta.ema(trading_df['close'], length=20*390)
    #trading_df['ema_50_day'] = ta.ema(trading_df['close'], length=50*390)
    #trading_df['ema_100_day'] = ta.ema(trading_df['close'], length=100*390)

    return trading_df.fillna(0)

def prepare_state_df(tickers, data_path):
    df_list = []
    column_list = []
    for ticker in tqdm(tickers):
        file = 'df_' + ticker + '_built.pkl'
        df = pd.read_pickle(data_path / file)
        trading_df = add_indicators(ticker, df)
        trading_df = trading_df.drop(columns=['status', 'timestamp'])
        #trading_df = trading_df.rename(columns={'close':ticker+'_close', 'high':ticker+'_high', 'low':ticker+'_low', 'open':ticker+'_open', 'volume':ticker+'_volume'})
        df_list.append(trading_df.to_numpy())
        column_list += list(trading_df.columns.values)

    #df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), df_list)
    df = pd.DataFrame(np.concatenate((df_list), axis=1), columns=column_list)

    return df

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