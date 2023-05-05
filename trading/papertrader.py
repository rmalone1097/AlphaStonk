import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ray.rllib.algorithms.algorithm import Algorithm
from envs.stock_env_trade import StockEnv
from utils.data_utils import *
from pathlib import Path

API_KEY = 'PKR994M0H9OM8NL6ZNNI'
SECRET_KEY = 'NfodkyTl7xkl2k6Fw1LfBiEAu3sXxQbqr1A5O888'

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

tickers = ['SPY', 'AAPL', 'AMZN', 'BAC', 'NVDA']
live_tickers = ['SPYlive', 'AAPLlive', 'AMZNlive', 'BAClive', 'NVDAlive']
# Notional value of every trade
notional_transaction_value = 10000
# Minute difference between start stamp and end stamp
minute_difference = 390 * 2 + 445

def fetch_old_data(tickers):
    end_stamp = math.floor(time.time())
    # Grab a week of data
    start_stamp = end_stamp - 10 * 24 * 60 * 60
    data_dir = Path.home() / 'data'

    dfs, _ = finnhub_data_writer(tickers=tickers,
                                      start_stamp=start_stamp,
                                      end_stamp=end_stamp,
                                      dir=data_dir)
    return dfs

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
    dfs = fetch_old_data(tickers)
    built_dfs = []
    for i, ticker in enumerate(tickers):
        filename = ticker + '_datastream.json'
        live_df = build_live_df(Path.home() / 'data' / filename)
        live_df = live_df[::-1]
        live_df = live_df.rename(columns={"close":"c", "max":"h", "min": "l", "open":"o", "s":"s", "time":"t", "volume":"v"})
        live_df = pd.concat([dfs[i], live_df])
        built_dfs.append(live_df)

    full_df, obs_df = prepare_live_df(tickers, built_dfs)
    env = StockEnv(config = {'full_df': full_df, 'obs_df': obs_df, 'tickers': tickers, 'print': False})
    obs = env.reset()
    action = algo.compute_single_action(obs)

'''market_order_data = MarketOrderRequest(
                    symbol = 
)'''