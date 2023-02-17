import numpy as np
from pathlib import Path
from data_utils import *

tickers = ['MSFT', 'AMZN', 'NVDA', 'WMT', 'AMD', 'BAC', 'GS', 'COST', 'XOM']
start_stamp = 928769400
end_stamp = 1676581140
data_path = Path.home() / 'data'

finnhub_data_writer(tickers, 928769400, 1676581140)
#for ticker in tickers:
#    df_builder(ticker, data_path / ticker + '_' + str(start_stamp) + '_' + str(end_stamp) + '_1_raw.pkl')
#df = finnhub_data_writer(['AAPL'], 928769400)
#df = df_builder(Path.home() / 'data' / 'AAPL_928769400_1676332664_1_raw.pkl')
#raw_df, df = df_builder('AAPL', Path.home() / 'data' / 'AAPL_928769400_1676581140_1_raw.pkl')
#df = pd.read_pickle(Path.home() / 'data' / 'AAPL_built.pkl')
#df['dt'] = df.apply(lambda row: datetime.fromtimestamp(row.timestamp), axis=1)          
#raw_df['dt'] = raw_df.apply(lambda row: datetime.fromtimestamp(row.t), axis=1)                                                                                                                                                                                                                                                                  