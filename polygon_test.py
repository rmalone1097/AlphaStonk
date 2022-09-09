import csv
import math
import time
import datetime
import os
import requests
import pandas as pd
import pytz

from scipy.interpolate import interp1d
from typing import cast
from urllib3 import HTTPResponse
from polygon import RESTClient
from tqdm import tqdm

API_KEY = 'jGYQQMOIgDQ9c3uefzYyi2AJLcqLXZzM'

# Date format YYYY-MM-DD
def write_data_to_file(ticker:str, multiplier:int, start_date:str, end_date:str, dir:str='', timespan:str='minute'):
    client = RESTClient(API_KEY)
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
            time = str(datetime.datetime.fromtimestamp(int(agg.timestamp/1000), pytz.timezone('US/Eastern')))[0:19]
            row = [time, agg.timestamp, agg.open, agg.high, agg.low, agg.close, agg.volume, agg.vwap, agg.transactions]
            writer.writerow(row)
    return path

def df_builder(path_to_csv:str):
    df = pd.read_csv(path_to_csv, header=0)
    return df

# Polygon outputs 5000 data points max about 5 days with pre/post data
def fetch_all_data(ticker:str, multiplier:int, start_date:str, end_date:str, timestamp:str='minute'):
    

file_path = write_data_to_file('AAPL', 1, '2022-05-19', '2022-05-20')
df = df_builder(file_path)