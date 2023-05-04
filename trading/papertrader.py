import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ray.rllib.algorithms.algorithm import Algorithm
from envs.stock_env_train import StockEnv
from utils.data_utils import *
from pathlib import Path

API_KEY = 'PKR994M0H9OM8NL6ZNNI'
SECRET_KEY = 'NfodkyTl7xkl2k6Fw1LfBiEAu3sXxQbqr1A5O888'

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

tickers = ['SPY', 'AAPL', 'AMZN', 'BAC', 'NVDA']
# Notional value of every trade
notional_transaction_value = 10000
# Minute difference between start stamp and end stamp
minute_difference = 390 * 2 + 445

def fetch_old_data(tickers):
    end_stamp = math.floor(time.time())
    # Grab a week of data
    start_stamp = end_stamp - 10 * 24 * 60 * 60
    data_dir = Path.home() / 'data'

    _, filepaths = finnhub_data_writer(tickers=tickers,
                                      start_stamp=start_stamp,
                                      end_stamp=end_stamp,
                                      dir=data_dir)
    return filepaths

def build_live_data(tickers, filepaths):
        
    for i in range(len(tickers)):
        df_builder(tickers[i], filepaths[i])

    full_state_df, obs_state_df, _, _ = prepare_state_df(tickers, data_dir, train_dps=390*2, test_dps=0, from_beginning=False)

    return full_state_df, obs_state_df

def check_account_value():
    account = trading_client.get_account()
    for property_name, value in account:
        print(f"\"{property_name}\": {value}")

#algo_path = Path.home() / 'ray_results'/'PPO'/'PPO_StockEnv_280fa_00000_0_2023-03-01_06-13-17'/'checkpoint_002500'
#algo = Algorithm.from_checkpoint(algo_path)

if __name__ == "__main__":
    old_df = pd.read_pickle(Path.home() / 'data' / 'AAPL_1682107263_1682971263_1_raw.pkl')

    live_df = build_live_df(Path.home() / 'data' / 'SPY_datastream.json')
    live_df = live_df[::-1]
    live_df = live_df.rename(columns={"close":"c", "max":"h", "min": "l", "open":"o", "s":"s", "time":"t", "volume":"v"})
    live_df = pd.concat([old_df, live_df])
    full_state_df, obs_state_df, _, _ = prepare_state_df(tickers, data_dir, train_dps=390*2, test_dps=0, from_beginning=False)

'''market_order_data = MarketOrderRequest(
                    symbol = 
)'''