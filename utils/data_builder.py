from data_utils import *

#fetch_all_data('SPY', 1, '2022-08-22', '2023-02-09')
df = df_builder(os.getcwd() + ' utils \\ SPY_minute_2022-08-22.csv')
filepath = os.getcwd() + '\\ utils \\SPY_minute_2022-08-22_built.pkl'
df.to_pickle(filepath)