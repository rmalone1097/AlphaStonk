import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import pandas_ta as ta
import mplfinance as mpf

from data_utils import add_indicators, plot_df_slice

pickle_dir = 'C:\\Users\\water\\Documents\\datasets\\stock_data'

df = pd.read_pickle(pickle_dir + '\\SPY_minute_2012-08-22_built.pkl')
trading_df = add_indicators(df)
plot_df_slice(trading_df, starting_index = 200000, ending_index = 230000)
