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
# Notional value of every trade
notional_transaction_value = 10000
# Minute difference between start stamp and end stamp
minute_difference = 390 * 2 + 445

algo_path = Path.home() / 'ray_results'/'PPO'/'PPO_StockEnv_280fa_00000_0_2023-03-01_06-13-17'/'checkpoint_002500'
algo = Algorithm.from_checkpoint(algo_path)

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

def check_account_value():
    account = trading_client.get_account()
    for property_name, value in account:
        print(f"\"{property_name}\": {value}")

if __name__ == "__main__":
    dfs = fetch_old_data(tickers)

    while True:
        built_dfs = []
        for i, ticker in enumerate(tickers):
            _, live_df = live_df_builder(tickers[i], dfs[i])
            built_dfs.append(live_df)
        
        full_df, obs_df = prepare_live_df(tickers, built_dfs)

        env = StockEnv(config = {'full_df': full_df, 'obs_df': obs_df, 'tickers': tickers, 'print': False, 'rew_function': 'energy'})
        obs = env.reset()
        action = algo.compute_single_action(obs)

        ticker_number = max(math.floor((action - 1) / 2), 0)
        if action == 0:
            order_side = None
        elif action % 2 == 1:
            order_side = OrderSide.BUY
        elif action % 2 == 0:
            order_side = OrderSide.SELL

        if order_side:
            market_order_data = MarketOrderRequest(
                                symbol = tickers[ticker_number],
                                notional = notional_transaction_value,
                                side = order_side,
                                time_in_force = TimeInForce.DAY
            )
            market_order = trading_client.submit_order(
                           order_data=market_order_data
            )
        
        if action != action_log and action_log != 0:
            TradingClient.close_all_positions()

        action_log = action
        ticker_log = tickers[ticker_number]
        time.sleep(60)