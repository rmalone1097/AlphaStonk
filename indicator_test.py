import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import pandas_ta as ta
import mplfinance as mpf

from data_utils import add_indicators, plot_df_slice

df = pd.read_pickle('SPY_minute_2012-08-22_built.pkl')
trading_df = add_indicators(df)
plot_df_slice(trading_df, starting_index = 200000, ending_index = 230000)
