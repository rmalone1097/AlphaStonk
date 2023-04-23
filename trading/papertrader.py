from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ray.rllib.algorithms.algorithm import Algorithm
from envs.stock_env_train import StockEnv

API_KEY = 'PKR994M0H9OM8NL6ZNNI'
SECRET_KEY = 'NfodkyTl7xkl2k6Fw1LfBiEAu3sXxQbqr1A5O888'

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

tickers = ['SPY', 'AAPL', 'BAC']
notional_transaction_value = 1000

account = trading_client.get_account()
for property_name, value in account:
    print(f"\"{property_name}\": {value}")

env = StockEnv(config = {'full_df': full_test_df, 'obs_df': obs_test_df, 'tickers': tickers, 'print': False})
algo_path = Path.home() / 'ray_results'/'PPO'/'PPO_StockEnv_280fa_00000_0_2023-03-01_06-13-17'/'checkpoint_002500'
algo = Algorithm.from_checkpoint(algo_path)

'''market_order_data = MarketOrderRequest(
                    symbol = 
)'''

