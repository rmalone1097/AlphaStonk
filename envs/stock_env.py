from gym import Env
from gym.spaces import Discrete, Box, Dict
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
        ### Action Space
        The action is a `ndarray` with shape `(1 + 2n,)` (short/long/none) where n is number of tickers
        | Num | Action                 |
        |-----|------------------------|
        | 0   | No position            |
        | 1   | Long position          |
        | 2   | Short position         |

        ### Observation Space
        Slice is a `ndarray` with shape `(390 * window_days,7n)` where n is the number of tickers and the elements correspond to the following:
        | Num | Observation                          | Min  | Max | Unit         |
        |-----|--------------------------------------|------|-----|--------------|
        | 0   | close                                | 0    | Inf | dollars ($)  |
        | 1   | high                                 | 0    | Inf | dollars ($)  |
        | 2   | low                                  | 0    | Inf | dollars ($)  |
        | 3   | open                                 | 0    | Inf | dollars ($)  |
        | 4   | volume                               | 0    | Inf | shares       |
        
        Vector is a 'ndarray' with shape '(5 + 19n,)' where n is the number of tickers and the elements correspond to the following:
        | Num | Observation                          | Min  | Max | Unit         |
        |-----|--------------------------------------|------|-----|--------------|
        | 0   | portfolio_value                      | -Inf | Inf | dollars ($)  |
        | 1   | position_log                         | 0    | 2   | discrete     |
        | 2   | action_taken                         | 0    | 2   | discrete     |
        | 3   | start_price                          | 0    | Inf | dollars ($)  |
        | 4   | holding_time                         | 0    | Inf | timesteps    |
        | 5   | latest_energy                        | -Inf | Inf | N/A          |
        | 6   | latest_close                         | 0    | Inf | dollars ($)  |
        | 7   | latest_high                          | 0    | Inf | dollars ($)  |
        | 8   | latest_low                           | 0    | Inf | dollars ($)  |
        | 9   | latest_open                          | 0    | Inf | dollars ($)  |
        | 10  | latest_volume                        | 0    | Inf | shares       |
        | 11  | latest_candle_counter                | 0    | Inf | candles      |
        | 12  | latest_ema_5                         | 0    | Inf | dollars ($)  |
        | 13  | latest_ema_10                        | 0    | Inf | dollars ($)  |
        | 14  | latest_ema_15                        | 0    | Inf | dollars ($)  |
        | 15  | latest_ema_25                        | 0    | Inf | dollars ($)  |
        | 16  | latest_ema_40                        | 0    | Inf | dollars ($)  |
        | 17  | latest_ema_65                        | 0    | Inf | dollars ($)  |
        | 18  | latest_ema_170                       | 0    | Inf | dollars ($)  |
        | 19  | latest_ema_250                       | 0    | Inf | dollars ($)  |
        | 20  | latest_ema_360                       | 0    | Inf | dollars ($)  |
        | 21  | latest_ema_445                       | 0    | Inf | dollars ($)  |
        '''
        # Number of tickers (dfs passed in initialization)
        self.num_tickers = config['num_tickers']
        self.df = config['df']
        self.action_space = Discrete(0 + 2*self.num_tickers)
        # Window width of data slice per step (days)
        self.window_days = 2
        # Observation dictionary
        self.observation_space = Dict({
            'slice': Box(low=0, high=np.inf, shape=(self.window_days*390, 5*self.num_tickers), dtype=np.float32),
            'vector': Box(low=np.concatenate((np.array([-np.inf, 0, 0, 0, 0], dtype=np.float32), np.repeat(np.concatenate((np.array([-np.inf], dtype=np.float32), np.zeros(16*self.num_tickers, dtype=np.float32))), self.num_tickers, axis=0))), 
                          high=np.concatenate((np.array([np.inf, 2, 2, np.inf, np.inf], dtype=np.float32), np.repeat(np.full(17, np.inf, dtype=np.float32), self.num_tickers, axis=0))))
        })
        self.data_tensor = self.df.to_numpy()
        # Num data points
        self.num_data = self.data_tensor.shape[0]
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
        # Current price list of all tickers
        self.current_price_list = []
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
        # Energy (5-15 EMA cloud difference) used for reward
        self.energy = 0
        # Episode length. Should be rollout length (for algos with rollout) * some scalar
        self.ep_timesteps = 2048 * 5
        # Tracks net worth to put in vector (Markov property?)
        self.portfolio = 0
        # For use in portfolio calculation
        self.transaction_value = 1000

    def step(self, action):
        assert self.state is not None, "Call reset before using step method"

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

        full_slice = self.data_tensor[first_idx:last_idx, :]
        assert full_slice.shape[0] == last_idx - first_idx, "Full Slice is failing"
        self.state['slice'] = full_slice
        self.current_price_list = [self.state['slice'][-1, 5*i] for i in range(self.num_tickers)]
        self.current_price = self.current_price_list[math.floor(self.position_log / 2)]

        # Worth of position, calculated as percentage change
        if self.position_log == 0:
            position_value = 0
        elif self.position_log % 2 == 1:
            position_value = (self.current_price - self.start_price) / self.start_price * 100
        elif self.position_log % 2 == 2:
            position_value = (self.start_price - self.current_price) / self.start_price * 100
        
        # Energy, defined as difference between EMA_25 and EMA_170. Daily candle counter used in reward calculation
        latest_close = self.current_price
        latest_daily_candle = full_slice[-1, 7]
        latest_ema_25 = full_slice[-1, 11]
        latest_ema_170 = full_slice[-1, 14]
        self.energy = (latest_ema_25 - latest_ema_170) / latest_ema_170 * 100

        # Reward calculation, defined as energy + slope of EMA_25 with some additional weight
        if latest_daily_candle > 120 or latest_daily_candle == 1:
            reward = self.energy + ((latest_close - latest_ema_25) / latest_ema_25 * 250)
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
        
        self.portfolio += self.transaction_value * position_value

        vector = np.array([self.portfolio, self.energy, self.position_log, action, self.start_price, self.holding_time])
        last_dp = full_slice[-1, :]
        self.state['vector'] = np.concatenate((vector, last_dp), axis=0)

        # Close old position and open new one
        if self.position_log != action:

            # Position taken, add 1 to position count
            self.num_positions += 1

            # Calcualte final ROI and update total
            self.roi = position_value
            self.total_roi += self.roi

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

            # Start price of new position is the current price
            self.start_price = self.current_price
            self.holding_time = self.minimum_holding_time
        
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

        # The state of the environment is the data slice that the agent will have access to to make a decision
        slice_tensor_list = []
        vector_tensor_list = []
        for i in range(self.num_tickers):
            slice_tensor_list.append(self.tensor_list[i][start_idx:end_idx, :])
            vector_tensor_list.append(self.tensor_list[i][end_idx, :])

        #TODO: Transactions is spelled wrong in the df
        self.state = {'slice': np.vstack(slice_tensor_list), 
        'vector': np.concatenate((np.zeros(6, dtype=np.float32), np.hstack(vector_tensor_list)))}
        #print(self.state['vector'])
        self.current_price_list = [tensor[-1, 0] for tensor in slice_tensor_list]
        self.start_price = self.current_price_list[0]
        self.state_idx = [start_idx, end_idx]
        return self.state