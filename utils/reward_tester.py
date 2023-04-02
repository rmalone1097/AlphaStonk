import matplotlib.pyplot as plt
#import mpmath

from data_utils import *
from pathlib import Path

pickle_dir = Path.home()
df = pd.read_pickle(Path.home() / 'data' / 'df_SPY_built.pkl')
trading_df = add_indicators('SPY', df)
trading_df = trading_df.fillna(0)

#plot_energy_cloud(trading_df, starting_index=1950, ending_index=2400)
difference = 2048
starting_index = 700000
ending_index =starting_index + difference
plot_energy_cloud('SPY', trading_df, starting_index=starting_index, ending_index=ending_index)