import numpy as np
from pathlib import Path
from data_utils import *

tickers = ['AAPL']
#tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'WMT', 'AMD', 'BAC', 'GS', 'COST', 'OXY']
cande_length = 1
start_stamp = 928761600
end_stamp = 1676581140
data_path = Path.home() / 'data'

#finnhub_data_writer(tickers, start_stamp, end_stamp)
#tickers = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'WMT', 'AMD', 'BAC', 'GS', 'COST', 'OXY']
#SPY_df = prepare_state_df(['SPY'], data_path)
#df = finnhub_data_writer(['AAPL'], 928769400)
#df = df_builder(Path.home() / 'data' / 'AAPL_928769400_1676332664_1_raw.pkl')
df_builder('SPY', Path.home() / 'data' / 'SPY_928761600_1676581140_1_raw.pkl')
df_builder('AAPL', Path.home() / 'data' / 'AAPL_928761600_1676581140_1_raw.pkl')
SPY_df = pd.read_pickle(Path.home() / 'data' / 'df_SPY_built.pkl')
AAPL_df = pd.read_pickle(Path.home() / 'data' / 'df_AAPL_built.pkl')

counter = 0

'''for i, row in tqdm(enumerate(AAPL_df.itertuples(index=True)), total=len(AAPL_df)):
    if row.Index != SPY_df.iloc[i].name:
        counter += 1
    else:
        counter = 0
    
    if counter >= 10:
        print('AAPL: ', row.Index)
        print('SPY: ', SPY_df.iloc[i].name)'''

#df['dt'] = df.apply(lambda row: datetime.fromtimestamp(row.timestamp), axis=1)         
'''raw_df = pd.read_pickle(Path.home() / 'data' / 'SPY_928769400_1676581140_1_raw.pkl')  
raw_df['dt'] = raw_df.apply(lambda row: datetime.fromtimestamp(row.t), axis=1)         
SPY_df = prepare_state_df(['SPY'], data_path)''' 

# df builder loop
'''for ticker in tickers:
    file_name = ticker + '_' + str(start_stamp) + '_' + str(end_stamp) + '_' + str(cande_length) + '_raw.pkl'
    df, raw_df = df_builder(ticker, data_path / file_name)'''