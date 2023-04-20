import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import pandas_ta as ta
import mplfinance as mpf

from pathlib import Path
from data_utils import add_indicators, plot_df_slice

pickle_dir = Path.home() / 'data'

df = pd.read_pickle(pickle_dir / 'df_SPY_built.pkl')
trading_df = add_indicators('SPY', df)
#plot_df_slice(trading_df, starting_index = 200000, ending_index = 230000)
