import pandas as pd
import numpy as np
import os
import plotly.express as px

from DataHandler import DataHandler
from StatArbStrategy import StatArbStrategy


class Trade:
    def __init__(self, window, start, finish, dir='data', coins_file='coin_universe_150K_40.csv',
                 prices_file='coin_all_prices.csv'):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010]
        by looping from start time to finish time.
        :param window: the time window-size
        :param start: the start datetime
        :param finish: the finish datetime
        :param dir: directory name to save data
        :param coins_file: the coins csv filename
        :param prices_file: the prices csv filename
        """
        # Initialize variables
        # Reads in all the data to be used by the trading classes
        self.data = DataHandler(coins_file=os.path.join(dir, coins_file), prices_file=os.path.join(dir, prices_file))
        self.window = window
        self.start = start
        self.finish = finish
        self.dir = dir
        self.coins_file = coins_file
        self.prices_file = prices_file
        self.strategy = StatArbStrategy(window, start, finish, self.data.symbols_df, self.data.st_ret_df)
        self.results_dict = None  # Contains all the results from the strategy
        self.trade_decisions = None
        self.SR = None
        self.MDD1 = None
        self.MDD2 = None
        self.egn_port1_rets = None
        self.egn_port2_rets = None

    def run_strategy(self):
        """
        A function to run the trading strategy and store the results
        :return: None
        """
        self.results_dict = self.strategy.get_trading_signals(self.start, self.finish, self.window)
        self.trade_decisions = self.trade(self.results_dict['signals'])

    def get_ret(self, df):
        """
        A function to get the hourly returns
        :param df: prices dataframe
        :return: the dataframe percentage change hour to hour
        """
        return df.loc[self.start:self.finish].pct_change()[1:]

    def get_prices(self, df):
        """
        A function to obtain the hourly prices given a start and finish datetime
        :param df: dataframe of prices
        :return: sliced dataframe
        """
        return df.loc[self.start:self.finish]

    def trade(self, signals):
        """
        A function to map trading signals to trading weights
        :param signals: A dataframe of trading signals
        :return: trading signals mapped to weights (1 is long, -1 is short, 0 no trade)
        """
        ts = pd.DataFrame(np.where(signals == "BTO",
                                   1,
                                   np.where(signals == "STO",
                                            -1,
                                            0)),
                          index=signals.index,
                          columns=signals.columns)
        return ts

    def get_weighted_prices(self):
        """
        A function that calculates the proportion of prices in each row
        :return: the weighted prices
        """
        return self.prices_df / self.prices_df.sum(axis=0)

    def get_strategy_ret(self):
        """
        A function that computes the strategy return
        :return: The strategy return at each hour
        """
        return (self.weighted_ret * self.get_weighted_prices()).sum(axis=1)

    # Getting the cumulative and average returns of the portfolio
    def get_strat_port_returns(self, decisions, rets):
        """
        A function to obtain the strategy portfolio returns
        :param decisions: trading weights
        :param rets: return series
        :return: portfolio returns
        """
        rets = rets[decisions.columns][
            (self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]
        positions = rets.multiply(decisions)
        average_ret = positions.mean(axis=1)
        cum_rets = (1 + average_ret).cumprod() - 1
        return pd.Series(average_ret, index=decisions.index), pd.Series(cum_rets, index=decisions.index)

    def get_eigen_port_returns(self, Q, rets):
        """
        A function that computes the eigenportfolio returns
        :param Q: The eigenportfolio weights
        :param rets: the return series
        :return: the eigenportfolio returns
        """
        Q = Q.dropna(axis=1, how='all').fillna(0)
        Q.index.name = 'startTime'
        rets = rets[Q.columns].fillna(0)
        rets.columns = Q.columns
        return pd.Series(Q.multiply(rets).sum(axis=1), index=Q.index)

    def results_to_csv(self):
        """
        A function that saves results to a csv file
        :return: None
        """
        directory = 'results'
        for key in self.results_dict:
            self.results_dict[key].to_csv(os.path.join(directory, key + '.csv'))

    def plot(self, df, title, special=False, start_time="2021-09-26 00:00:00", finish_time="2021-10-25 23:00:00"):
        """
        A function that plots a given series
        :param df: data to plot
        :param title: title of the plot
        :param special: set to True if plotting BTC and ETH s_scores
        :param start_time: start datetime for s_scores plot
        :param finish_time: end datetime for s_scores plot
        :return: None
        """
        if not special:
            fig = px.line(df, title=title)
        else:
            cpy = df.loc[start_time:finish_time].copy(deep=True)
            fig = px.line(cpy, title=title)
        fig.write_image(title + ".png")
        fig.show()

    def plot_multiple(self, df, title):
        """
        A function that handles multiple plots
        :param df: data to plot
        :param title: title of the plot
        :return: None
        """
        fig = px.line(df, x=df.index, y=df.columns, title=title)
        fig.write_image(title + ".png")
        fig.show()

    def cumulative_stategy_plot(self, cum_rets):
        """
        A function to plot the cumulative strategy
        :param cum_rets: cumulative return series
        :return: None
        """
        self.plot(cum_rets, 'Cumulative Strategy Returns')

    def stategy_rets_hist(self, strat_rets):
        """
        A function to plot a histogram of the return series
        :param strat_rets: strategy returns
        :return: None
        """
        fig = px.histogram(strat_rets)
        fig.write_image("Histogram-Returns.png")
        fig.show()

    def eig_portfolio_ret_plot(self, title):
        """
        A function that plots the eigenportfolio return series
        :param title: title of the plot
        :return: None
        """
        rets_df = self.data.returns_df[
            (self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]
        self.egn_port1_rets = self.get_eigen_port_returns(self.results_dict['eigportwgts1'], rets_df)
        self.egn_port2_rets = self.get_eigen_port_returns(self.results_dict['eigportwgts2'], rets_df)
        cumwrets1 = (1 + self.egn_port1_rets).cumprod() - 1
        cumwrets2 = (1 + self.egn_port2_rets).cumprod() - 1
        cumbitrets = (1 + self.data.returns_df['BTC'][
            (self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]).cumprod() - 1
        cumethrets = (1 + self.data.returns_df['ETH'][
            (self.data.returns_df.index >= self.start) & (self.data.returns_df.index <= self.finish)]).cumprod() - 1
        cum_ret_df = pd.concat([cumwrets1, cumwrets2, cumbitrets, cumethrets], axis=1)
        self.plot_multiple(cum_ret_df, title)

    def get_SR(self, ret_series, rf: float = 0.0):
        """
        Compute Sharpe Ratio.
        :return: Sharpe Ratio
        """
        return (ret_series.mean() * np.sqrt(8760) - rf) / ret_series.std()

    def get_MDD(self, portfolio_value: pd.DataFrame, window=12, min_periods=1):
        """
        Compute maximum draw-down (MDD).
        :return: MDD
        """
        # Change min_periods if you want to let the first X days data have an expanding window
        max_rolling_window = portfolio_value.rolling(window, min_periods=min_periods).max()
        DD = portfolio_value / max_rolling_window - 1.0
        return DD.rolling(window, min_periods=min_periods).min()

    def report_results(self):
        """
        A function that reports the results to console, saves files to csv, and plots needed data.
        :return: None
        """
        # Writing data to csv
        self.results_to_csv()

        # Plotting the eigen portfolio weights
        self.plot(self.results_dict['eigportwgts1'].loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False),
                  'Eigen Portfolio 1 Weights at T1')
        self.plot(self.results_dict['eigportwgts1'].loc['2022-04-15 20:00:00'].dropna().sort_values(ascending=False),
                  'Eigen Portfolio 1 Weights at T2')
        self.plot(self.results_dict['eigportwgts2'].loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False),
                  'Eigen Portfolio 2 Weights at T1')
        self.plot(self.results_dict['eigportwgts2'].loc['2022-04-15 20:00:00'].dropna().sort_values(ascending=False),
                  'Eigen Portfolio 2 Weights at T2')

        # Plotting the cumulative returns of the eigen portfolios relative to bitcoin and etherium
        self.eig_portfolio_ret_plot('Cumulative Returns Plot')

        # Plotting the S-Scores of bitcoin and etherium
        self.plot(self.results_dict['score']['BTC'], 'Bitcoin S-Scores', special=True)
        self.plot(self.results_dict['score']['ETH'], 'Etherium S-Scores', special=True)

        # Plotting the returns for the trading strategy
        rets, cum_rets = self.get_strat_port_returns(self.trade_decisions, self.data.returns_df)
        self.cumulative_stategy_plot(cum_rets)
        self.stategy_rets_hist(rets)

        # Compute metrics
        self.SR = self.get_SR(rets)
        print(f"The Sharpe Ratio is = {self.SR}")
        self.MDD1 = self.get_MDD(self.egn_port1_rets)
        self.MDD2 = self.get_MDD(self.egn_port2_rets)
        self.plot(self.MDD1, "Maximum Draw Down of Portfolio 1 Returns")
        self.plot(self.MDD2, "Maximum Draw Down of Portfolio 2 Returns")
