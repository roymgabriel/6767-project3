import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Factors():
   def __init__(self, window, start, finish, coins_file='data/coin_universe_150K_40.csv', prices_file='data/coin_all_prices.csv'):
      # Reading in symbols for the period
      self.symbols_df = pd.read_csv(coins_file)
      self.symbols_df = self.__clean_dates(self.symbols_df)
      self.symbols_df = self.symbols_df[(self.symbols_df.index >= start) & (self.symbols_df.index <= finish)]

      # Reading in prices from csv
      self.prices_df = pd.read_csv(prices_file)
      self.prices_df = self.__clean_dates(self.prices_df)
      self.prices_df = self.__clean_data(self.prices_df)

      # Getting the returns from the prices dataframe
      self.returns_df = self.__get_rets(self.prices_df)

      # Additional parameters
      self.M = window
      self.start = start
      self.finish = finish

   def __clean_dates(self, df):
      df['startTime'] = pd.to_datetime(df['startTime'].apply(lambda x : x.split(':')[0]), format='%Y-%m-%dT%H')
      df = df.set_index('startTime').drop('time', axis=1)
      return df

   def __clean_data(self, df):
      df.ffill(inplace=True)
      return df

   def __get_rets(self, prices_df):
      ret_df = prices_df.pct_change()
      return ret_df

   def __standardize_rets(self, df):
      scaler = StandardScaler()
      scaler.fit(df)
      return pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

   def __get_corr_mat(self, df):
      st_rets = self.__standardize_rets(df)
      return st_rets.corr()

   def __pca(self, df):
      emp_corr = self.__get_corr_mat(df)
      pca_model = PCA(n_components=2)
      pca_model.fit(df)
      # Returns the eigen vectors
      return pca_model.transform(df)

   def test_pca(self, time):
      used_symbols = list(self.symbols_df.loc[time])
      time_idx = self.returns_df.index.get_loc(time)
      st_rets = self.__standardize_rets(self.returns_df[used_symbols].iloc[time_idx - self.M : time_idx - 1])
      return self.__pca(st_rets)


def main():
   test = Factors(240, '021-03-08T05 :00 :00+00 :00', '022-09-25T23 :00 :00+00 :00')
   test.symbols_df.head()

if __name__ == "__main__":
   main()