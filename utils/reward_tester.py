import matplotlib.pyplot as plt
import mpmath

from data_utils import *
from pathlib import Path

pickle_dir = Path.home()
df = pd.read_pickle(pickle_dir / 'SPY_minute_2012-08-22_built_gcp.pkl')
trading_df = add_indicators(df)
trading_df = trading_df.fillna(0)

#plot_energy_cloud(trading_df, starting_index=1950, ending_index=2400)
difference = 2048
starting_index = 642000
ending_index =starting_index + difference
plot_energy_cloud(trading_df, starting_index=starting_index, ending_index=ending_index)