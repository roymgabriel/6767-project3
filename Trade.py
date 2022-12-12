import pandas as pd
import numpy as np

from StatArbStrategy import StatArbStrategy


class Trade:
    def __init__(self, window, start, finish, coins_file='data/coin_universe_150K_40.csv',
                prices_file='data/coin_all_prices.csv'):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010]
        by looping from start time to finish time
        :param
        :param
        :return:
        """
        # Initialize variables
        self.window = window
        self.start = start
        self.finish = finish
        self.coins_file = coins_file
        self.prices_file = prices_file
        self.trading_signals = {}
        self.hourly_returns = {}
        self.weighted_returns = {}

        # create date range in hourly offset
        self.date_range = pd.date_range(start, finish, freq='H')

    def get_state(self):
        """
        A function to get all necessary info at time t
        :return:
        """
        pass

    def get_trading_signals(self):
        """
        Get trading signals at each time t from start to finish
        :return:
        """
        for start_time in self.date_range:
            arb = StatArbStrategy(window=self.window,
                                  start=start_time,
                                  finish=self.finish,
                                  coins_file=self.coins_file,
                                  prices_file=self.prices_file)

            params = arb.get_params()
            s = arb.get_s_score(params)

            # get trading signals with a certain start time and their returns
            self.trading_signals[start_time] = arb.generate_trading_signals(s)
            self.hourly_returns[start_time] = arb.hourly_returns

        return self.trading_signals


    def map_signal_to_trade(self, x):
        if x == "BTO":
            return 1
        elif x == "STO":
            return -1
        elif x == "CSP":
            return 1
        elif x == "CLP":
            return -1

    def trade(self, pos, ):
        # TODO: Need to map whether there was a change in signal or not from previous step
        # TODO: The hourly returns do not match date, need to fix that but the concept is there
        for start_time, df in self.trading_signals.items():
            df['trade_pos'] = df.apply(self.map_signal_to_trade)
            self.weighted_returns[start_time] = df['trade_pos'] * self.hourly_returns[start_time]





