import pandas as pd
import numpy as np
import os
import plotly.express as px

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
        self.window = window
        self.start = start
        self.finish = finish
        self.dir = dir
        self.coins_file = coins_file
        self.prices_file = prices_file
        self.ret_df = None
        self.prices_df = None

        # create date range in hourly offset
        self.date_range = pd.date_range(start, finish, freq='H')

        self.trading_signals, self.egn_port_wgts1, self.egn_port_wgts2, self.egn_port_vect1, self.egn_port_vect2, self.s_score = self.get_trading_signals()
        self.weighted_ret = self.trade()
        self.portfolio_ret = self.get_portfolio_ret()

    def get_ticker_symbols(self):
        return list(pd.read_csv(self.prices_file, nrows=1).columns[2:])

    def get_trading_signals(self):
        """
        Get trading signals at each time t from start to finish
        :return:
        """
        i = 0
        cols = []
        trading_signals = {}
        egn_portfolio_wgts1 = []
        egn_portfolio_wgts2 = []
        egn_vect1 = []
        egn_vect2 = []
        s_score = []
        for start_time in self.date_range:
            print(start_time)
            arb = StatArbStrategy(window=self.window,
                                  start=start_time,
                                  finish=self.finish,
                                  coins_file=os.path.join(self.dir, self.coins_file),
                                  prices_file=os.path.join(self.dir, self.prices_file))

            # get trading signals with a certain start time and their returns
            trading_signals[start_time] = arb.generate_trading_signals()
            if i == 0:
                self.ret_df = self.get_ret(arb.prices_df)
                self.prices_df = self.get_prices(arb.prices_df)
            i += 1

            cols.append(arb.prices_df.columns)
            # Getting the eigen portfolio and eigen portfolio vectors for each hour of the top two eigen vectors
            egn_portfolio_wgts1.append(arb.factors.Q_j.iloc[0,:].rename(start_time))
            egn_vect1.append(arb.factors.pca_eigenvectors.iloc[0,:].rename(start_time))
            egn_portfolio_wgts2.append(arb.factors.Q_j.iloc[1,:].rename(start_time))
            egn_vect2.append(arb.factors.pca_eigenvectors.iloc[1,:].rename(start_time))
            s_score.append(arb.s_score.rename(start_time))

        cols = np.unique(cols)
        self.ret_df = self.ret_df.loc[:, cols].shift(-1)
        self.prices_df = self.prices_df.loc[:, cols]
        trading_signals = pd.DataFrame.from_dict(trading_signals, orient='index')
        egn_port_df1 = pd.DataFrame(egn_portfolio_wgts1, columns=cols)
        egn_vect_df1 = pd.DataFrame(egn_vect1, columns=cols)
        egn_port_df2 = pd.DataFrame(egn_portfolio_wgts2, columns=cols)
        egn_vect_df2 = pd.DataFrame(egn_vect2, columns=cols)
        s_score_df = pd.DataFrame(s_score, columns = cols)
        return trading_signals, egn_port_df1, egn_port_df2, egn_vect_df1, egn_vect_df2, s_score_df

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
        return self.prices_df / self.prices_df.sum(axis=0)

    def get_portfolio_ret(self):
        return (self.weighted_ret * self.get_weighted_prices()).sum(axis=1)

    # Funciton to write needed results to csv
    def results_to_csv(self):
        directory = 'results'
        self.egn_port_wgts1.to_csv(os.path.join(directory, 'Eigen_Porfolio_Weights_1.csv'))
        self.egn_port_wgts2.to_csv(os.path.join(directory, 'Eigen_Porfolio_Weights_2.csv'))
        self.egn_port_vect1.to_csv(os.path.join(directory, 'Eigen_Porfolio_Vectors_1.csv'))
        self.egn_port_vect2.to_csv(os.path.join(directory, 'Eigen_Porfolio_Vectors_2.csv'))
        self.trading_signals.to_csv(os.path.join(directory, 'Trading_Signals.csv'))

    def plot(self, df, title):
        fig = px.line(df, title=title)
        fig.show()

    def plot_multiple(self, df, title):
        fig = px.line(df, x=df.index, y=df.columns, title=title)
        fig.show()


    def cumulative_ret_plot(self, title):
        wrets1 = pd.Series(np.sum((self.egn_port_wgts1.shift().dropna(how='all').fillna(0).values * self.ret_df.fillna(0).values), axis=1), name='WR1', index=self.ret_df.index)
        wrets2 = pd.Series(np.sum((self.egn_port_wgts2.shift().dropna(how='all').fillna(0).values * self.ret_df.fillna(0).values), axis=1), name='WR2', index=self.ret_df.index)
        cumwrets1 = (1 + wrets1).cumprod() - 1
        cumwrets2 = (1 + wrets2).cumprod() - 1
        cumbitrets = (1 + self.ret_df['BTC']).cumprod() - 1
        cumethrets = (1 + self.ret_df['ETH']).cumprod() - 1
        cum_ret_df = pd.concat([cumwrets1, cumwrets2, cumbitrets, cumethrets], axis=1)
        self.plot_multiple(cum_ret_df, title)


def main():
    start = "2021-09-26 00:00:00"
    finish = "2022-09-25 23:00:00"#"2022-09-25 23:00:00"

    trade = Trade(window=240, start=start, finish=finish)
    trade.results_to_csv()
    trade.plot(trade.egn_port_wgts1.loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 1 Weights at T1')
    trade.plot(trade.egn_port_wgts2.loc['2021-09-26 12:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 1 Weights at T1')
    trade.plot(trade.egn_port_wgts1.loc['2021-10-25 23:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 2 Weights at T2')
    trade.plot(trade.egn_port_wgts2.loc['2021-10-25 23:00:00'].dropna().sort_values(ascending=False), 'Eigen Portfolio 2 Weights at T2')
    trade.cumulative_ret_plot('Cumulative Returns Portfolio')
    trade.plot(trade.s_score['BTC'], 'Bitcoin S-Scores')
    trade.plot(trade.s_score['ETH'], 'Etherium S-Scores')

    
    # print(trade.weighted_ret)


if __name__ == "__main__":
    main()
