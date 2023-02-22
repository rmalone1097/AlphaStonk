import numpy as np
from pathlib import Path
from data_utils import *

tickers = ['MSFT', 'AMZN', 'NVDA', 'WMT', 'AMD', 'BAC', 'GS', 'COST', 'OXY']
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'data'

finnhub_data_writer(tickers, start_stamp, end_stamp)
#tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'WMT', 'AMD', 'BAC', 'GS', 'COST', 'OXY']
#SPY_df = prepare_state_df(['SPY'], data_path)
#df = finnhub_data_writer(['AAPL'], 928769400)
#df = df_builder(Path.home() / 'data' / 'AAPL_928769400_1676332664_1_raw.pkl')
#raw_df, df = df_builder('SPY', Path.home() / 'data' / 'SPY_928761600_1676581140_1_raw.pkl')
#df = pd.read_pickle(Path.home() / 'data' / 'AAPL_built.pkl')
#df['dt'] = df.apply(lambda row: datetime.fromtimestamp(row.timestamp), axis=1)         
'''raw_df = pd.read_pickle(Path.home() / 'data' / 'SPY_928769400_1676581140_1_raw.pkl')  
raw_df['dt'] = raw_df.apply(lambda row: datetime.fromtimestamp(row.t), axis=1)         
SPY_df = prepare_state_df(['SPY'], data_path)'''                                                                                                                                                                                                                                               