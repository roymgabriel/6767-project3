import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Factors:
    def __init__(self, window, start, finish, coins_file='data/coin_universe_150K_40.csv',
                 prices_file='data/coin_all_prices.csv'):
        # Reading in symbols for the period
        self.symbols_df = pd.read_csv(coins_file)
        self.symbols_df = self.__clean_dates(self.symbols_df)
        self.symbols_df = self.symbols_df[(self.symbols_df.index >= start) & (self.symbols_df.index <= finish)]

        # Reading in prices from csv
        self.prices_df = pd.read_csv(prices_file)
        self.prices_df = self.__clean_dates(self.prices_df)
        self.prices_df = self.__clean_data(self.prices_df)
        self.prices_df = self.prices_df[self.prices_df.index <= finish]

        # Getting the returns from the prices dataframe
        self.returns_df = self.__get_rets(self.prices_df)

        # Additional parameters
        self.M = window
        self.start = start
        self.finish = finish
        self.asset_std = None
        self.eigenvalues = None
        self.hourly_rets = None

        self.Q_j = self.get_Q()
        self.F_jk = self.get_factor_return()

    def __clean_dates(self, df):
        df['startTime'] = pd.to_datetime(df['startTime'].apply(lambda x: x.split(':')[0]), format='%Y-%m-%dT%H')
        df = df.set_index('startTime').drop('time', axis=1)
        return df

    def __clean_data(self, df):
        df.ffill(inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df

    def __get_rets(self, prices_df):
        """
      Compute factor returns of the two risk factors at time t.
      :return:
      """
        ret_df = prices_df.pct_change()
        return ret_df

    def get_standardize_rets(self, df):
        self.asset_std = df.std()  # could also do np.sqrt(scaler.var_)
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    def get_corr_mat(self, df):
        st_rets = self.get_standardize_rets(df)
        return st_rets.corr()

    def get_pca(self, df):
        emp_corr = self.get_corr_mat(df)
        pca_model = PCA(n_components=2)
        pca_model.fit(emp_corr)
        self.eigenvalues = pca_model.explained_variance_
        eigenvectors = pd.DataFrame(pca_model.components_, columns=df.columns)
        # Returns the eigen vectors
        return eigenvectors

    def get_Q(self):
        used_symbols = list(self.symbols_df.loc[self.start])
        time_idx = self.returns_df.index.get_loc(self.start)
        self.hourly_rets = self.returns_df[used_symbols].iloc[time_idx - self.M: time_idx]
        st_rets = self.get_standardize_rets(self.hourly_rets).dropna(axis=0)
        pca_eigenvectors = self.get_pca(st_rets)
        Q = pca_eigenvectors / self.asset_std
        return Q

    def get_factor_return(self):
        # Q is 2x40 and hourly_rets 239x40
        return self.Q_j @ self.hourly_rets.T


def main():
    test = Factors(240, '2021-03-08 05:00:00', '2022-09-25 23:00:00')
    print(test.symbols_df.head())


if __name__ == "__main__":
    main()
