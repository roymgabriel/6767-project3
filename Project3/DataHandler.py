import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self, coins_file='data/coin_universe_150K_40.csv',
                 prices_file='data/coin_all_prices.csv'):
        """
        A class that reads the crypto token csv files, cleans them, and calculates the returns needed
        for the Statistical Arbitrage Trading strategy.
        :param coins_file: the coins csv filename
        :param prices_file: the prices csv filename
        """
        # Reading in symbols for the period
        self.symbols_df = pd.read_csv(coins_file)
        self.symbols_df = self.__clean_dates(self.symbols_df)
        self.symbols_df = self.__add_eth(self.symbols_df)  # Adding etherium to each row
        # self.symbols_df = self.symbols_df[(self.symbols_df.index >= start) & (self.symbols_df.index <= finish)]

        # Reading in prices from csv
        self.prices_df = pd.read_csv(prices_file)
        self.prices_df = self.__clean_dates(self.prices_df)
        self.prices_df = self.__clean_data(self.prices_df)
        # self.prices_df = self.prices_df[self.prices_df.index <= finish]

        # Getting the returns from the prices dataframe
        self.returns_df = self.__get_rets(self.prices_df)
        self.returns_df = self.__clean_data(self.returns_df)

        self.st_ret_df = self.__get_standardize_rets(self.returns_df)
        self.st_ret_df = self.__clean_data(self.st_ret_df)

    # Function to add etherium if it hasn't been included
    def __add_eth(self, symbols_df):
        """
        A function that adds ETH if it has not been included as a top 40 token based on market cap.
        :param symbols_df: the coins symbols dataframe
        :return: the cleaned symbols dataframe including ETH
        """
        mask = (symbols_df == "ETH").sum(axis=1)
        symbols_df['39'] = symbols_df['39'].mask(mask == 0).fillna("ETH")
        return symbols_df

    def __clean_dates(self, df):
        """
        A function that cleans the dates
        :param df: dataframe of any data
        :return: cleaned dates dataframe
        """
        df['startTime'] = pd.to_datetime(df['startTime'].apply(lambda x: x.split(':')[0]), format='%Y-%m-%dT%H')
        df = df.set_index('startTime').drop('time', axis=1)
        return df

    def __clean_data(self, df):
        """
        A function that cleans the data by forward fill and replacing infinite values with 0.
        :param df: dataframe of any values
        :return: cleaned dataframe
        """
        df.ffill(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.fillna(0)

    def __get_rets(self, prices_df):
        """
        A function that computes factor returns of the two risk factors at time t.
        :param prices_df: prices dataframe
        :return: returns of the prices
        """
        ret_df = prices_df.pct_change()
        return ret_df

    def __get_standardize_rets(self, df):
        """
        A function that standardizes the return series.
        :param df: return price series
        :return: standardized returns
        """
        self.asset_std = df.std()  # could also do np.sqrt(scaler.var_)
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
