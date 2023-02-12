from pathlib import Path
from data_utils import *

path_dir = Path.home() / 'Git' /  'AlphaStonk' / 'utils'

fetch_all_data('UVXY', 1, '2022-08-22', '2023-02-09')
df = df_builder(path_dir / 'VIX_minute_2022-08-22.csv')
#filepath = path_dir / 'SPY_minute_2022-08-22_built.pkl'
#df.to_pickle(filepath)