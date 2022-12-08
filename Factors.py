import pandas as pd
import numpy as np

class Factors():
   def __init__(self, window, start, finish, coins_file='coin_universe_150K_40.csv', prices_file='coin_all_prices.csv'):
      # Reading in symbols for the period
      self.symbols_df = pd.read_csv(coins_file)
      self.symbols_df = self.__clean_dates(self.symbols_df)
      self.symbols_df = self.symbols_df[(self.symbols_df.index >= start) & (self.symbols_df.index <= finish)]

      # Reading in prices from csv
      self.prices_df = pd.read_csv(prices_file)
      self.symbols_df = self.__clean_dates(self.prices_df)
   
      # Additional parameters
      self.M = window
      self.start = start
      self.finish = finish

   def __clean_dates(self, df):
      df['startTime'] = pd.to_datetime(df['startTime'].apply(lambda x : x.split(':')[0]), format='%Y-%m-%dT%H')
      df = df.set_index('startTime').drop('time', axis=1)
      return df


   def get_rets(self):
      pass


def main():
   test = Factors(240, '021-03-08T05 :00 :00+00 :00', '022-09-25T23 :00 :00+00 :00')
   test.symbols_df.head()

if __name__ == "__main__":
   main()