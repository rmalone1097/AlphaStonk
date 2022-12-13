from gym import Env
from gym.spaces import Discrete, Box
from typing import Optional
import datetime

import torch
import numpy as np
import pandas as pd
from data_utils import *

class StockEnv(Env):
    def __init__(self, df):
        '''
        ### Action Space
        The action is a `ndarray` with shape `(1,)` which can take values in range # tickers * 0 (short/long/none)
        | Num | Action                 |
        |-----|------------------------|
        | 0   | No position            |
        | 1   | Long position          |
        | 2   | Short position         |

        ### Observation Space
        The observation is a `ndarray` with shape `(candles,featuers)` where the elements correspond to the following:
        | Num | Observation                          | Min  | Max | Unit         |
        |-----|--------------------------------------|------|-----|--------------|
        | 0   | open                                 | 0    | Inf | dollars ($)  |
        | 1   | high                                 | 0    | Inf | dollars ($)  |
        | 2   | low                                  | 0    | Inf | dollars ($)  |
        | 3   | close                                | 0    | Inf | dollars ($)  |
        | 4   | volume                               | 0    | Inf | shares       |
        | 5   | vwap                                 | 0    | Inf | dollars ($)  | 
        | 6   | transactions                         | 0    | Inf | transactions |
        | 7   | daily candle counter                 | 0    | Inf | candles      |
        | 8   | ema_5                                | 0    | Inf | dollars ($)  |
        | 9   | ema_10                               | 0    | Inf | dollars ($)  |
        | 10  | ema_15                               | 0    | Inf | dollars ($)  |
        | 11  | ema_25                               | 0    | Inf | dollars ($)  |
        | 12  | ema_40                               | 0    | Inf | dollars ($)  |
        | 13  | ema_65                               | 0    | Inf | dollars ($)  |
        | 14  | ema_170                              | 0    | Inf | dollars ($)  |
        | 15  | ema_250                              | 0    | Inf | dollars ($)  |
        | 16  | ema_360                              | 0    | Inf | dollars ($)  |
        | 17  | ema_445                              | 0    | Inf | dollars ($)  |
        | 18  | ema_900                              | 0    | Inf | dollars ($)  |
        | 19  | ema_1000                             | 0    | Inf | dollars ($)  |
        | 20  | ema_5_day                            | 0    | Inf | dollars ($)  |
        | 21  | ema_10_day                           | 0    | Inf | dollars ($)  |
        | 22  | ema_20_day                           | 0    | Inf | dollars ($)  |
        | 23  | ema_50_day                           | 0    | Inf | dollars ($)  |
        | 24  | ema_100_day                          | 0    | Inf | dollars ($)  |
        '''
        self.action_space = Discrete(3)
        # Window width of data slice per step (days)
        self.window_days = 5
        # Number of candles by number of featuers
        # TODO add previous action and holding time to state
        self.observation_space = Box(low=0, high=np.inf, shape=(self.window_days*390,25), dtype=np.float16)
        self.df = df
        # Every transcation to have this value ($)
        self.transaction_value = 1000
        # Net worth to track cumulative reward
        self.net_worth = 0
        # Variable to keep track of initial underlying at start of position
        self.start_price = 1
        # Observed state (data slice)
        self.state = None
        # List of indexes for ease of data frame iteration
        self.state_idx = []
        # Variable to keep track of position between steps
        self.position_log = 0
        # Log to keep track of trades
        self.trade_log = []
        # Timestep length to update action
        self.timestep = 1
        # Logged value representing amount of long positions
        self.longs = 0
        # Logged value representing amount of short positions
        self.shorts = 0
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
        # Current price
        self.current_price = 0
        # Minimum time (in minutes) a position must be held
        self.minimum_holding_time = 1
        # Holding time for a position
        self.holding_time = 0
        # Action log
        self.action_log = 0
        # Dictionary of open-close price and PL (unused)
        self.pl_dict = {}
        # Positive if winning streak, negative if losing streak
        self.streak = 0
        # Hold time of position
        self.holding_time = 0
        # Defined as the point at which a trade with a positive position value will yield 0 reward due to decay
        self.decay_factor = 1000
        # Track ROI
        self.roi = 0
        # Total ROI to compute average
        self.total_roi = 0
        # Total number of positions to compute ROI average
        self.num_positions = 0
        # Average ROI
        self.average_roi = 0

    def step(self, action):
        assert self.state is not None, "Call reset before using step method"

        # Step data window 1 candle
        # Fetch first and last index of the window and add 1
        first_idx, last_idx = self.state_idx[0] + 1, self.state_idx[1] + 1
        if last_idx + self.timestep >= len(self.df):
            self.wins = 0
            self.losses = 0
            self.longs = 0
            self.shorts = 0
            self.total_roi = 0
            self.num_positions = 0
            self.position_log = 0
            action = 0
            done = True
        else:
            done = False

        # If data point after last is after market close, find the next market open point
        if self.df.iloc[last_idx]['daily_candle_counter'] == 0:
            for i, row in enumerate(self.df.iloc[last_idx:].itertuples()):
                # add i to last_idx and first idx to keep slice length consistent
                if row.daily_candle_counter != 0:
                    first_idx, last_idx = first_idx + i, last_idx + i
                    break

        df_slice = self.df.iloc[first_idx:last_idx]
        self.state = df_slice.loc[:, 'open':].to_numpy()
        self.current_price = df_slice.iloc[-1]['close']

        # Worth of position, calculated as percentage change
        if self.position_log == 1:
            position_value = (self.current_price - self.start_price) / self.start_price * 100
        elif self.position_log == 2:
            position_value = (self.start_price - self.current_price) / self.start_price * 100
        else:
            position_value = 0

        # Close old position and open new one
        if self.position_log != action:

            # Position taken, add 1 to position count
            self.num_positions += 1

            # Calcualte final ROI and update total
            self.roi = position_value
            self.total_roi += self.roi

            # Agent closed position so position value is final. Can be used to tally win/loss
            if position_value > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Maintains streak, which is logged but currently unused
            if position_value > 0 and self.streak >= 0:
                self.streak += 1
            elif position_value < 0 and self.streak <= 0:
                self.streak -= 1
            else:
                self.streak = 0

            '''if self.position_log == 1 and self.reward < 0:
                self.reward = self.reward * 1.5

            if self.position_log == 2 and self.reward < 0:
                self.reward = self.reward * 1.5'''
            
            # Skip amount of canldes specified by timestep once a position is taken
            if action != 0:
                first_idx += self.minimum_holding_time
                last_idx += self.minimum_holding_time
            
            # Count longs and shorts
            if action == 1:
                self.longs += 1
            elif action == 2:
                self.shorts += 1

            # Start price of new position is the current price
            self.start_price = self.current_price
            self.holding_time = self.minimum_holding_time

            '''
        # If holding no position, slight penalty equal to 1% loss per day DEPRICATED
        elif self.position_log == 0:
            percentage_multiplier = 0.01
            steps_in_trading_day = 390
            self.reward = -self.transaction_value * percentage_multiplier / steps_in_trading_day
            '''

        # Posiiton is held. Grant reward based on reward function and position value
        else:
            if position_value < 0:
                self.reward = (-position_value - (-position_value * self.holding_time) / self.decay_factor) + position_value*2
            else:
                self.reward = position_value - (position_value * self.holding_time) / self.decay_factor

            self.holding_time += 1

        if self.num_positions != 0:
            self.win_ratio = self.wins / (self.wins + self.losses)
            self.long_ratio = self.longs / (self.longs + self.shorts)
            self.average_roi = self.total_roi / self.num_positions
        self.position_log = action
        info = {}
        self.action = action
        self.state_idx = [first_idx, last_idx]

        return self.state, self.reward, done, info

    def render(self):
        pass

    def reset(self):
        # Search through dataframe and look for first data point where market is open
        first_found = False
        first_trading_stamp = 0

        for i, row in enumerate(self.df.itertuples()):
            if row.daily_candle_counter != 0 and first_found == False:
                first_found = True

                # First market open data point
                first_valid_day = datetime.datetime.fromtimestamp(int(row.timestamp) / 1000, pytz.timezone('US/Eastern'))

                # Name, or set index of df, is expected to be timestep (milliseconds) and will be used to locate data points of interest
                first_valid_name = i

                # Calculation of first trading date, window days + 1
                first_trading_day = first_valid_day + timedelta(days=self.window_days + 1)

                # If Saturday or Sunday, get it to Monday
                #TODO: i set this to 3 and 2 to fix a labor day bug. Remember that this should be 2 and 1
                if first_trading_day.weekday() == 5:
                    first_trading_day += timedelta(days=3)
                elif first_trading_day.weekday() == 6:
                    first_trading_day += timedelta(days=2)
                assert (first_trading_day.hour, first_trading_day.minute) == (9, 30), "Calculation of first trading point is incorrect"

                # Calculation of first trading point on first trading day (9:30AM EST on first trading day)
                first_trading_stamp = int(round(first_trading_day.timestamp() * 1000))

            if row.timestamp == first_trading_stamp:
                first_trading_name = i
                break

        # The state of the environment is the data slice that the agent will have access to to make a decision
        df_slice = self.df.iloc[first_valid_name:first_trading_name]
        self.state = df_slice.loc[:, 'open':].to_numpy()
        self.current_price = self.state[9, 3]
        self.start_price = self.current_price
        self.state_idx = [first_valid_name, first_trading_name]
        return self.state