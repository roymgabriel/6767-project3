import pandas as pd
import numpy as np
import os
import plotly.express as px

from data_handler import data_handler
from StatArbStrategy import StatArbStrategy


class Trade:
    def __init__(self, window, start, finish, dir='data', coins_file='coin_universe_150K_40.csv',
                 prices_file='coin_all_prices.csv'):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010]
        by looping from start time to finish time
        :param
        :param
        :return:
        """
        # Initialize variables
        # Reads in all of the data to be used by the trading classes
        self.data = data_handler(coins_file=os.path.join(dir, coins_file), prices_file=os.path.join(dir, prices_file))
        self.window = window
        self.start = start
        self.finish = finish
        self.dir = dir
        self.coins_file = coins_file
        self.prices_file = prices_file
        self.strategy = StatArbStrategy(window, start, finish, self.data.symbols_df, self.data.st_ret_df)
        self.results_dict = None # Contains all of the results from the strategy
        self.trade_decisions = None


    def run_strategy(self):
        self.results_dict = self.strategy.get_trading_signals(self.start, self.finish, self.window)
        self.trade_decisions = self.trade(self.results_dict['signals'])

    def get_ret(self, df):
        return df.loc[self.start:self.finish].pct_change()[1:]

    def get_prices(self, df):
        return df.loc[self.start:self.finish]

    def trade(self, signals):
        ts = pd.DataFrame(np.where(signals == "BTO",
                                   1,
                                   np.where(signals == "STO",
                                            -1,
                                            0)),
                          index=signals.index,
                          columns=signals.columns)
        return ts

    def get_weighted_prices(self):
        return self.prices_df / self.prices_df.sum(axis=0)

    def get_strategy_ret(self):
        return (self.weighted_ret * self.get_weighted_prices()).sum(axis=1)

    #Getting the cumulative and average returns of teh portfolio
    def get_strat_port_returns(self, decisions, rets):
        rets = rets[decisions.columns][(self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]
        positions = rets.multiply(decisions)
        average_ret = positions.mean(axis=1)
        cum_rets = (1 + average_ret).cumprod() - 1
        return pd.Series(average_ret, index=decisions.index), pd.Series(cum_rets, index=decisions.index)

    def get_eigen_port_returns(self, Q, rets):
        Q = Q.dropna(axis=1, how='all').fillna(0)
        Q.index.name = 'startTime'
        rets = rets[Q.columns].fillna(0)
        rets.columns = Q.columns
        return pd.Series(Q.multiply(rets).sum(axis=1), index=Q.index)

    # Funciton to write needed results to csv
    def results_to_csv(self):
        directory = 'results'
        for key in self.results_dict:
            self.results_dict[key].to_csv(os.path.join(directory, key + '.csv'))

    def plot(self, df, title):
        fig = px.line(df, title=title)
        fig.show()

    def plot_multiple(self, df, title):
        fig = px.line(df, x=df.index, y=df.columns, title=title)
        fig.show()

    def cumulative_stategy_plot(self, cum_rets):
        self.plot(cum_rets, 'Cumulative Strategy Returns')

    def stategy_rets_hist(self, strat_rets):
        fig = px.histogram(strat_rets)
        fig.show()

    def eig_portfolio_ret_plot(self, title):
        rets_df = self.data.returns_df[(self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]
        egn_port1_rets = self.get_eigen_port_returns(self.results_dict['eigportwgts1'], rets_df)
        egn_port2_rets = self.get_eigen_port_returns(self.results_dict['eigportwgts2'], rets_df)
        cumwrets1 = (1 + egn_port1_rets).cumprod() - 1
        cumwrets2 = (1 + egn_port2_rets).cumprod() - 1
        cumbitrets = (1 + self.data.returns_df['BTC'][(self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]).cumprod() - 1
        cumethrets = (1 + self.data.returns_df['ETH'][(self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]).cumprod() - 1
        cum_ret_df = pd.concat([cumwrets1, cumwrets2, cumbitrets, cumethrets], axis=1)
        self.plot_multiple(cum_ret_df, title)

    def report_results(self):
        # Writing data to csv
        self.results_to_csv()

        # Plotting the eigen portfolio weights
        self.plot(self.results_dict['eigportwgts1'].loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 1 Weights at T1')
        self.plot(self.results_dict['eigportwgts1'].loc['2022-04-15 20:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 1 Weights at T2')
        self.plot(self.results_dict['eigportwgts2'].loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 2 Weights at T1')
        self.plot(self.results_dict['eigportwgts2'].loc['2022-04-15 20:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 2 Weights at T2')

        # Plotting the cumulative returns of the eigen portfolios relative to bitcoin and etherium
        self.eig_portfolio_ret_plot('Cumulative Returns Plot')

        # Plotting the S-Scores of bitcoin and etherium
        self.plot(self.results_dict['score']['BTC'], 'Bitcoin S-Scores')
        self.plot(self.results_dict['score']['ETH'], 'Etherium S-Scores')

        # Plotting the returns for the trading strategy
        rets, cum_rets = self.get_strat_port_returns(self.trade_decisions, self.data.returns_df)
        self.cumulative_stategy_plot(cum_rets)
        self.stategy_rets_hist(rets)


def main():
    start = "2021-09-26 00:00:00"
    finish = "2022-09-25 23:00:00"#"2022-09-25 23:00:00"

    trade = Trade(window=240, start=start, finish=finish)
    trade.run_strategy()
    trade.report_results()


if __name__ == "__main__":
    main()
