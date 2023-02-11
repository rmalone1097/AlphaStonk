from pathlib import Path
from data_utils import *

path_dir = Path.home() / 'Git' /  'AlphaStonk' / 'utils'

#fetch_all_data('SPY', 1, '2022-08-22', '2023-02-09')
path_dir = Path.home() / 'Git' /  'AlphaStonk' / 'utils'
df = df_builder(path_dir / 'SPY_minute_2022-08-22.csv')
filepath = path_dir / 'SPY_minute_2022-08-22_built.pkl'
df.to_pickle(filepath)