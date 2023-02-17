import numpy as np
from pathlib import Path
from data_utils import *

#finnhub_data_writer(['OXY'], 928769400)
#df = finnhub_data_writer(['AAPL'], 928769400)
#df = df_builder(Path.home() / 'data' / 'AAPL_928769400_1676332664_1_raw.pkl')
raw_df, df = df_builder('AAPL', Path.home() / 'data' / 'AAPL_928769400_1676332664_1_raw.pkl')
#df = pd.read_pickle(Path.home() / 'data' / 'AAPL_built.pkl')
df['dt'] = df.apply(lambda row: datetime.fromtimestamp(row.timestamp), axis=1)          
raw_df['dt'] = raw_df.apply(lambda row: datetime.fromtimestamp(row.t), axis=1)                                                                                                                                                                                                                                                                  