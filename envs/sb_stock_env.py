from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
from typing import Optional
import datetime
from ray.rllib.env.env_context import EnvContext

import numpy as np
import pandas as pd
from utils.data_utils import *
import random

random.seed(10)

class StockEnv(Env):
    def __init__(self, config: EnvContext):
        '''
        ### Input
        The input is a `dataframe` with shape `(candles,1 + 16n)` where n is the number of tickers and the elmeents correspond to the following:
        | Num     | Observation                          | Min  | Max | Unit         |
        |---------|--------------------------------------|------|-----|--------------|
        | 0       | daily_candle_counter                 | 0    | 390 | candles      |

        | 1(n+1)  | energy                               | -Inf | Inf | N/A          |
        | 2(n+1)  | close                                | 0    | Inf | dollars ($)  |
        | 3(n+1)  | high                                 | 0    | Inf | dollars ($)  |
        | 4(n+1)  | low                                  | 0    | Inf | dollars ($)  |
        | 5(n+1)  | open                                 | 0    | Inf | dollars ($)  |
        | 6(n+1)  | volume                               | 0    | Inf | shares       |
        | 7(n+1)  | ema_5                                | 0    | Inf | dollars ($)  |
        | 8(n+1)  | ema_10                               | 0    | Inf | dollars ($)  |
        | 9(n+1)  | ema_15                               | 0    | Inf | dollars ($)  |
        | 10(n+1) | ema_25                               | 0    | Inf | dollars ($)  |
        | 11(n+1) | ema_40                               | 0    | Inf | dollars ($)  |
        | 12(n+1) | ema_65                               | 0    | Inf | dollars ($)  |
        | 13(n+1) | ema_170                              | 0    | Inf | dollars ($)  |
        | 14(n+1) | ema_250                              | 0    | Inf | dollars ($)  |
        | 15(n+1) | ema_360                              | 0    | Inf | dollars ($)  |
        | 16(n+1) | ema_445                              | 0    | Inf | dollars ($)  |

        ### Action Space
        The action is a `ndarray` with shape `(1 + 2n,)` (short/long/none) where n is number of tickers
        | Num     | Action                 |
        |---------|------------------------|
        | 0       | No position            |

        | 1(n+1)  | Long position          |
        | 2(n+1)  | Short position         |

        ### Observation Space
        Slice is a `ndarray` with shape `(390 * window_days,5n)` where n is the number of tickers and the elements correspond to the following:
        | Num | Observation                          | Min  | Max | Unit         |
        |-----|--------------------------------------|------|-----|--------------|
        | 0   | close                                | 0    | Inf | dollars ($)  |
        | 1   | high                                 | 0    | Inf | dollars ($)  |
        | 2   | low                                  | 0    | Inf | dollars ($)  |
        | 3   | open                                 | 0    | Inf | dollars ($)  |
        | 4   | volume                               | 0    | Inf | shares       |
        
        Vector is a 'ndarray' with shape '(6 + 16n,)' where n is the number of tickers and the elements correspond to the following:
        | Num     | Observation                          | Min  | Max | Unit         |
        |---------|--------------------------------------|------|-----|--------------|
        | 0       | portfolio_value                      | -Inf | Inf | dollars ($)  |
        | 1       | position_log                         | 0    | 2   | discrete     |
        | 2       | action_taken                         | 0    | 2   | discrete     |
        | 3       | start_price                          | 0    | Inf | dollars ($)  |
        | 4       | holding_time                         | 0    | Inf | timesteps    |
        | 5       | latest_candle_counter                | 0    | Inf | candles      |

        | 6(n+1)  | latest_energy                        | -Inf | Inf | N/A          |
        | 7(n+1)  | latest_close                         | 0    | Inf | dollars ($)  |
        | 8(n+1)  | latest_high                          | 0    | Inf | dollars ($)  |
        | 9(n+1)  | latest_low                           | 0    | Inf | dollars ($)  |
        | 10(n+1) | latest_open                          | 0    | Inf | dollars ($)  |
        | 11(n+1) | latest_volume                        | 0    | Inf | shares       |
        | 12(n+1) | latest_ema_5                         | 0    | Inf | dollars ($)  |
        | 13(n+1) | latest_ema_10                        | 0    | Inf | dollars ($)  |
        | 14(n+1) | latest_ema_15                        | 0    | Inf | dollars ($)  |
        | 15(n+1) | latest_ema_25                        | 0    | Inf | dollars ($)  |
        | 16(n+1) | latest_ema_40                        | 0    | Inf | dollars ($)  |
        | 17(n+1) | latest_ema_65                        | 0    | Inf | dollars ($)  |
        | 18(n+1) | latest_ema_170                       | 0    | Inf | dollars ($)  |
        | 19(n+1) | latest_ema_250                       | 0    | Inf | dollars ($)  |
        | 20(n+1) | latest_ema_360                       | 0    | Inf | dollars ($)  |
        | 21(n+1) | latest_ema_445                       | 0    | Inf | dollars ($)  |
        '''
        self.print_config = config['print']
        self.tickers = config['tickers']
        # Number of tickers (dfs passed in initialization)
        self.num_tickers = len(self.tickers)
        # Full df input
        self.full_df = config['full_df']
        # Obs df is only columns used in state slice
        self.obs_df = config['obs_df']
        self.action_space = Discrete(1 + 2*self.num_tickers)
        # Window width of data slice per step (days)
        self.window_days = 2
        # Observation dictionary
        self.observation_space = Dict({
            'slice': Box(low=0, high=np.inf, shape=(self.window_days*390, 5*self.num_tickers), dtype=np.float32),
            'vector': Box(low=np.concatenate((np.array([-np.inf, 0, 0, 0, 0, 0], dtype=np.float32), np.tile(np.concatenate((np.array([-np.inf], dtype=np.float32), np.zeros(15, dtype=np.float32))), self.num_tickers))), 
                          high=np.concatenate((np.array([np.inf, 2*self.num_tickers, 2*self.num_tickers, np.inf, np.inf, np.inf], dtype=np.float32), np.repeat(np.full(16, np.inf, dtype=np.float32), self.num_tickers, axis=0))))
        })
        self.full_df_tensor = self.full_df.to_numpy()
        self.obs_df_tensor = self.obs_df.to_numpy()
        # Num data points
        self.num_data = self.full_df_tensor.shape[0]
        # Variable to keep track of initial underlying at start of position
        self.start_price = 1
        # Observed state (data slice)
        self.state = None
        # List of indexes for ease of data frame iteration
        self.state_idx = []
        # Variable to keep track of position between steps
        self.position_log = 0
        # Keeps track of timestep during training
        self.timestep = 1
        # Logged value representing amount of long positions
        self.longs = 0
        # Logged value representing amount of short positions
        self.shorts = 0
        # Logged value representing candles passed with no position
        self.zeros = 0
        # Logged value representing amount of zeros to total amount of candles
        self.zero_ratio = 0
        # Logged value representing ratio of long positions to total positions
        self.long_ratio = 0
        # Logged value representing number of positive trades
        self.wins = 0
        # Logged value representing number of negative trades
        self.losses = 0
        # logged value defined as wins over total trades
        self.win_ratio = 0
        # Reward
        self.reward = 0
        # Action
        self.action = 0
        # Current price list of ticker position is in
        self.current_price = 0
        # Minimum time (in minutes) a position must be held
        self.minimum_holding_time = 0
        # Holding time for a position
        self.holding_time = 0
        # Positive if winning streak, negative if losing streak
        self.streak = 0
        # Average holding time
        self.average_holding_time = 0
        # Total holding time used for average holding time calculation
        self.total_holding_time = 0
        # Track ROI
        self.roi = 0
        # Total ROI to compute average
        self.total_roi = 0
        # Total number of positions to compute ROI average
        self.num_positions = 1
        # Average ROI
        self.average_roi = 0
        # Number of minutes of long positions used to calculate zero ratio
        self.long_candles = 0
        # Number of minutes of short positions used to calculate zero ratio
        self.short_candles = 0
        # Cumulative ROI for long positions
        self.long_roi = 0
        # Cumulative ROI for short positions
        self.short_roi = 0
        # Average ROI for long positions
        self.average_long_roi = 0
        # Average ROI for short positions
        self.average_short_roi = 0
        # Episode length. Should be rollout length (for algos with rollout) * some scalar
        self.ep_timesteps = 2048 * 5
        # Tracks net worth to put in vector (Markov property?)
        self.portfolio = 0
        # For use in portfolio calculation
        self.transaction_value = 1000

    def step(self, action):
        assert self.state is not None, "Call reset before using step method"

        ''' State update block '''

        # Step data window 1 candle
        # Fetch first and last index of the window and add 1
        first_idx, last_idx = self.state_idx[0] + 1, self.state_idx[1] + 1
        if self.timestep == self.ep_timesteps:
            done = True
        else:
            done = False

        # Environment is never seeing points before or after market close
        '''# While data point after last is after market close, add one until next market open point
        while self.data_tensor[last_idx, 7] == 0:
            first_idx, last_idx = first_idx + 1, last_idx + 1'''

        full_slice = self.full_df_tensor[first_idx:last_idx, :]
        assert full_slice.shape[0] == last_idx - first_idx, "Full Slice is failing"

        self.state['slice'] = self.obs_df_tensor[first_idx:last_idx, :]

        ''' Reward calculation block '''

        # Ticker number starting at 0 of last position (position log)
        self.ticker_number = max(math.floor((self.position_log - 1) / 2), 0)
        self.current_price = self.state['slice'][-1, 5*(self.ticker_number)]

        # Worth of position, calculated as percentage change
        if self.position_log == 0:
            position_value = 0
        elif self.position_log % 2 == 1:
            position_value = (self.current_price - self.start_price) / self.start_price * 100
        elif self.position_log % 2 == 0:
            position_value = (self.start_price - self.current_price) / self.start_price * 100
        
        # Energy, defined as difference between EMA_25 and EMA_170. Daily candle counter used in reward calculation
        latest_close = self.current_price
        latest_daily_candle = full_slice[-1, 0]
        latest_energy = full_slice[-1, 16*self.ticker_number + 1]
        latest_ema_25 = full_slice[-1, 16*self.ticker_number + 10]
        latest_ema_170 = full_slice[-1, 16*self.ticker_number + 13]
        
        # Reward calculation, defined as energy + slope of EMA_25 with some additional weight
        if latest_daily_candle > 120 or latest_daily_candle == 1:
            reward = latest_energy + ((latest_close - latest_ema_25) / latest_ema_25 * 250)
        else:
            reward = (latest_close - latest_ema_25) / latest_ema_25 * 250
        
        # Reward setting
        if self.position_log == 1:
            self.reward = reward
        elif self.position_log == 2:
            self.reward = -reward
        elif self.position_log == 0:
            if abs(reward) <= 0.4 or latest_daily_candle < 15:
                self.reward = 0
            else:
                self.reward = -abs(reward)
        
        ''' State vector update block '''

        vector = np.array([self.portfolio, self.position_log, action, self.start_price, self.holding_time])
        last_dp = full_slice[-1, :]
        self.state['vector'] = np.concatenate((vector, last_dp), axis=0)

        if self.print_config:
            print('action: ', action)
            print('position log: ', self.position_log)
            print('Ticker number of last position: ', self.ticker_number)
            print('Position value: ', position_value)
            print('Reward: ', self.reward)
            print('Latest daily candle: ', latest_daily_candle)
            print('Start price: ', self.start_price)
            print('Latest close: ', latest_close)
            print('Latest ema 25: ', latest_ema_25)
            print('Latest ema 170', latest_ema_170)
            print('Latest energy: ', latest_energy)
            print('Total ROI: ', self.total_roi)
            print('')

        ''' New action update block '''

        # Close old position and open new one
        if self.position_log != action:

            # Position taken, add 1 to position count
            self.num_positions += 1

            # Calcualte final ROI and update total
            self.roi = position_value
            self.total_roi += self.roi

            self.portfolio += max(self.transaction_value, self.portfolio) * (self.roi / 100)

            if self.position_log == 1:
                self.long_roi += self.roi
            elif self.position_log == 2:
                self.short_roi += self.roi

            # Agent closed position so position value is final. Can be used to tally win/loss
            if position_value > 0:
                self.wins += 1
            elif position_value < 0:
                self.losses += 1
            
            # Maintains streak, which is logged but currently unused
            if position_value > 0 and self.streak >= 0:
                self.streak += 1
            elif position_value < 0 and self.streak <= 0:
                self.streak -= 1
            else:
                self.streak = 0
            
            '''# Skip amount of canldes specified by timestep once a position is taken
            if action != 0:
                first_idx += self.minimum_holding_time
                last_idx += self.minimum_holding_time'''
            
            # Count longs and shorts
            if action == 1:
                self.longs += 1
            elif action == 2:
                self.shorts += 1

            new_ticker_number = max(math.floor((action - 1) / 2), 0)
            self.start_price = self.state['slice'][-1, 5*(new_ticker_number)]
            self.holding_time = self.minimum_holding_time
        
        ''' Logging calculation block '''
        
        # Count long and short candles
        if action == 1:
            self.long_candles += 1
        elif action == 2:
            self.short_candles += 1
        elif action == 0:
            self.zeros += 1

        if action != 0:
            self.holding_time += 1
            self.total_holding_time += 1
        
        self.win_ratio = self.wins / self.num_positions
        self.long_ratio = self.longs / self.num_positions
        self.zero_ratio = self.zeros / (self.long_candles + self.short_candles + self.zeros)
        self.average_roi = self.total_roi / self.num_positions
        self.average_holding_time = self.total_holding_time / self.num_positions
        self.average_long_roi = self.long_roi / (self.longs + 1)
        self.average_short_roi = self.short_roi / (self.shorts + 1)

        self.position_log = action
        info = {}
        self.action = action
        self.state_idx = [first_idx, last_idx]
        self.timestep += 1

        return self.state, self.reward, done, info

    def render(self):
        pass

    def reset(self):

        self.timestep = 0
        self.wins = 0
        self.losses = 0
        self.longs = 0
        self.shorts = 0
        self.long_candles = 0
        self.short_candles = 0
        self.zeros = 1
        self.long_roi = 0
        self.short_roi = 0
        self.total_roi = 0
        self.num_positions = 1
        self.position_log = 0
        self.total_holding_time = 0
        self.portfolio = 0

        # Finds random point in the data to start from
        start_idx = random.randrange(self.num_data - self.ep_timesteps - self.window_days * 390 - 1)
        end_idx = start_idx + self.window_days * 390

        #TODO: Transactions is spelled wrong in the df
        self.state = {'slice': self.obs_df_tensor[start_idx:end_idx, :], 
                      'vector': np.zeros(6 + 16*self.num_tickers, dtype=np.float32)}
        #print(self.state['vector'])
        self.start_price = self.state['slice'][-1, 0]
        self.state_idx = [start_idx, end_idx]
        return self.state