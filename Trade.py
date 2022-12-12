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
        self.ret_df = None
        self.prices_df = None

        # create date range in hourly offset
        self.date_range = pd.date_range(start, finish, freq='H')

        self.trading_signals = self.get_trading_signals()
        self.weighted_ret = self.trade()
        self.portfolio_ret = self.get_portfolio_ret()

    def get_trading_signals(self):
        """
        Get trading signals at each time t from start to finish
        :return:
        """
        i = 0
        cols = []
        trading_signals = {}
        for start_time in self.date_range:
            print(start_time)
            arb = StatArbStrategy(window=self.window,
                                  start=start_time,
                                  finish=self.finish,
                                  coins_file=self.coins_file,
                                  prices_file=self.prices_file)

            params = arb.get_params()
            s = arb.get_s_score(params)

            # get trading signals with a certain start time and their returns
            trading_signals[start_time] = arb.generate_trading_signals(s)
            if i == 0:
                self.ret_df = self.get_ret(arb.prices_df)
                self.prices_df = self.get_prices(arb.prices_df)
            i += 1
            cols.append(arb.prices_df.columns)

        cols = np.unique(cols)
        self.ret_df = self.ret_df.loc[:, cols].shift(-1)
        self.prices_df = self.prices_df.loc[:, cols]
        trading_signals = pd.DataFrame.from_dict(trading_signals, orient='index')
        return trading_signals

    def get_ret(self, df):
        return df.loc[self.start:self.finish].pct_change()[1:]

    def get_prices(self, df):
        return df.loc[self.start:self.finish]

    def trade(self):
        ts = pd.DataFrame(np.where(self.trading_signals == "BTO",
                                   1,
                                   np.where(self.trading_signals == "STO",
                                            -1,
                                            0)),
                          index=self.trading_signals.index,
                          columns=self.trading_signals.columns)
        return ts * self.ret_df

    def get_weighted_prices(self):
        return self.prices_df / self.prices_df.sum(axis=1)

    def get_portfolio_ret(self):
        return self.weighted_ret * self.get_weighted_prices()


def main():
    start = "2021-09-26 00:00:00"
    finish = "2022-09-25 23:00:00"

    trade = Trade(window=240, start=start, finish=finish)
    print(trade.weighted_ret)


if __name__ == "__main__":
    main()
