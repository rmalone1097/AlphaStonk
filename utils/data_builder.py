import numpy as np
from pathlib import Path
from data_utils import *

#tickers = ['AAPL']
tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'AMD', 'BAC', 'COST', 'OXY']
cande_length = 1
start_stamp = 928761600
end_stamp = 1683732468
data_path = Path.home() / 'data'

finnhub_data_writer(tickers, start_stamp, end_stamp)

# df builder loop
for ticker in tickers:
    file_name = ticker + '_' + str(start_stamp) + '_' + str(end_stamp) + '_' + str(cande_length) + '_raw.pkl'
    df_builder(ticker, data_path / file_name)

SPY_df = pd.read_pickle(Path.home() / 'data' / 'df_SPY_built.pkl')
AAPL_df = pd.read_pickle(Path.home() / 'data' / 'df_AAPL_built.pkl')
MSFT_df = pd.read_pickle(Path.home() / 'data' / 'df_MSFT_built.pkl')
AMZN_df = pd.read_pickle(Path.home() / 'data' / 'df_AMZN_built.pkl')
NVDA_df = pd.read_pickle(Path.home() / 'data' / 'df_NVDA_built.pkl')
WMT_df = pd.read_pickle(Path.home() / 'data' / 'df_WMT_built.pkl')
AMD_df = pd.read_pickle(Path.home() / 'data' / 'df_AMD_built.pkl')
BAC_df = pd.read_pickle(Path.home() / 'data' / 'df_BAC_built.pkl')
GS_df = pd.read_pickle(Path.home() / 'data' / 'df_GS_built.pkl')
COST_df = pd.read_pickle(Path.home() / 'data' / 'df_COST_built.pkl')
OXY_df = pd.read_pickle(Path.home() / 'data' / 'df_OXY_built.pkl')

full_train_df, full_obs_df, _, _ = prepare_state_df(tickers, data_path, 2100000, 100000, from_beginning=False)