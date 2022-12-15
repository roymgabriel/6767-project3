import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class data_handler:
   def __init__(self, coins_file='data/coin_universe_150K_40.csv',
               prices_file='data/coin_all_prices.csv'):
      # Reading in symbols for the period
      self.symbols_df = pd.read_csv(coins_file)
      self.symbols_df = self.__clean_dates(self.symbols_df)
      self.symbols_df = self.__add_eth(self.symbols_df) # Adding etherium to each row
      #self.symbols_df = self.symbols_df[(self.symbols_df.index >= start) & (self.symbols_df.index <= finish)]

      # Reading in prices from csv
      self.prices_df = pd.read_csv(prices_file)
      self.prices_df = self.__clean_dates(self.prices_df)
      self.prices_df = self.__clean_data(self.prices_df)
      #self.prices_df = self.prices_df[self.prices_df.index <= finish]

      # Getting the returns from the prices dataframe
      self.returns_df = self.__get_rets(self.prices_df)
      self.returns_df = self.__clean_data(self.returns_df)

      self.st_ret_df = self.__get_standardize_rets(self.returns_df)
      self.st_ret_df = self.__clean_data(self.st_ret_df)


   # Function to add etherium if it hasnt been included
   def __add_eth(self, symbols_df):
      mask = symbols_df.iloc[-1].apply(lambda row: 'ETH' not in row, axis=1)
      print(mask)
      symbols_df['39'] = symbols_df['39'].mask(mask, 'ETH')
      return symbols_df

   # Clean the dates
   def __clean_dates(self, df):
      df['startTime'] = pd.to_datetime(df['startTime'].apply(lambda x: x.split(':')[0]), format='%Y-%m-%dT%H')
      df = df.set_index('startTime').drop('time', axis=1)
      return df

   # Clean the data
   def __clean_data(self, df):
      df.ffill(inplace=True)
      df.replace([np.inf, -np.inf], 0, inplace=True)
      return df.fillna(0)

   # Gets the returns
   def __get_rets(self, prices_df):
      """
   Compute factor returns of the two risk factors at time t.
   :return:
   """
      ret_df = prices_df.pct_change()
      return ret_df

   # Get the standardized returns
   def __get_standardize_rets(self, df):
      self.asset_std = df.std()  # could also do np.sqrt(scaler.var_)
      scaler = StandardScaler()
      return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)