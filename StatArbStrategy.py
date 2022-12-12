import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from Factors import Factors


class StatArbStrategy:
    def __init__(self, window, start, finish, coins_file='data/coin_universe_150K_40.csv',
                prices_file='data/coin_all_prices.csv'):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010].
        :param
        :param
        :return:
        """
        # Initialize variables
        self.window = window
        self.start = start
        self.finish = finish
        self.params = dict()

        # get Factor Returns
        factors = Factors(window=window, start=start, finish=finish)
        self.factor_returns = factors.get_factor_return()
        self.hourly_returns = factors.hourly_rets
        self.prices_df = factors.prices_df

        self.residuals = self.estimate_resid_returns()
        self.X_l, self.xl_residuals = self.get_X_l()
        self.params_df = self.get_params()


    def estimate_resid_returns(self):
        """
        Estimate residual returns of token S.
        :return:
        """
        regressor = LinearRegression()
        X = self.factor_returns  # 2x40
        y = self.hourly_returns  # 239x40
        d = {}
        for col in y.columns:
            model = regressor.fit(X.T, y[col])
            y_pred = model.predict(X.T)
            resid = y[col] - y_pred
            d[col] = resid
            self.params[col] = [model.intercept_] + model.coef_.tolist()  # b_0, b_1, b_2
        return pd.DataFrame(d)

    def get_X_l(self):
        """
        Function to get X_l in order to obtain a and b parameters
        :return:
        """
        # NOTE: this function should return parameters a and b for each asset
        X_l = self.residuals.cumsum()
        X_l_shift = X_l.shift(-1)
        regressor = LinearRegression()
        X = X_l[:-1]  # M-1 rows
        y = X_l_shift[:-1]  # M-1 rows
        coeffs = {}
        xl_residuals = {}
        for col in y.columns:
            model = regressor.fit(X[[col]], y[col])
            y_pred = model.predict(X[[col]])
            resid = y[col] - y_pred
            # Adding the residuals for a currency to a dictionary
            xl_residuals[col] = resid
            # Returns the a and b coefficients for each currency
            coeffs[col] = [model.coef_[0], model.intercept_]
        return pd.DataFrame.from_dict(coeffs, orient='index', columns=['b', 'a']).T, pd.DataFrame.from_dict(xl_residuals, orient='columns')



    def get_params(self):
        """
        A function to get greek parameters
        :return:
        """
        # I changed the function to loop through each currency and get the parameters for each one
        params = {}
        for col in self.X_l.columns:
            # Defining the input parameters for each currency
            a = self.X_l.loc['a', col]
            b = self.X_l.loc['b', col]
            resid = self.xl_residuals[col]

            kappa = -np.log(b) * 8760  # For the number of hours in a year
            m = a / (1-b)
            sigma = np.sqrt(np.var(resid) * 2 * kappa / (1 - b**2))
            sigma_eq = np.sqrt(np.var(resid) / (1 - b**2))
            params[col] = [kappa, m, sigma, sigma_eq]
        return pd.DataFrame.from_dict(params, orient='index', columns=['kappa', 'm', 'sigma', 'sigma_eq'])

    def get_s_score(self, params_df):
        """
        Calculate s-score at time t.
        :return:
        """
        cpy = params_df.copy(deep=True)
        m_mean = cpy['m'].mean()
        s_score = (m_mean - cpy['m']) / cpy['sigma_eq']
        return s_score

    def generate_trading_signals(self, s_score):
        """
        Generate trading signals at time t.
        :return:
        """
        #TODO: might need to pass these as paramters
        s_bo = 1.25
        s_so = 1.25
        s_bc = 0.75
        s_sc = 0.5

        # Conditions for trading signals
        conditions = [
            (s_score < -s_bo),
            (s_score > s_so),
            (s_score < s_bc) & (s_score >= -s_bo),
            (s_score > -s_sc) & (s_score <= s_so)
        ]

        # Mapped values
        values = ['BTO', 'STO', 'CSP', 'CLP']

        return pd.Series(np.select(conditions, values), name='Signals', index=s_score.index)

    def evaluate_strat(self):
        """
        Evaluate strategy performance over a testing period.
        :return:
        """
        pass

    def get_model_metrics(self):
        """
        Compute Sharpe Ratio and maximum draw-down (MDD).
        :return:
        """
        pass

    def get_SR(self, stats_df: pd.DataFrame, rf: float):
        """
        Compute Sharpe Ratio.
        :return:
        """
        return (stats_df.loc['mean'] - rf)/stats_df.loc['std']

    def get_MDD(self, df: pd.DataFrame, col: str, window=12, min_periods=1):
        """
        Compute maximum draw-down (MDD).
        :return:
        """
        # Change min_periods if you want to let the first X days data have an expanding window
        max_rolling_window = df[col].rolling(window, min_periods=1).max()
        DD = df[col]/max_rolling_window - 1.0
        return DD.rolling(window, min_periods=1).min()

    def plot_eigen_portfolio(self):
        """
        Plot the eigen-portfolio weights of the two eigen-portfolios at a certain time period.
        :return:
        """
        pass

    def save_trading_signals(self):
        """
        Create a csv file containing the trading signals of a given input.
        :return:
        """
        pass

def main():
    test = StatArbStrategy(window=240, start="2021-09-26 00:00:00", finish="2022-09-25 00:00:00")
    print(test.get_X_l()[1])
    params = test.get_params()
    print(params)
    s = test.get_s_score(params)
    print(s)
    print(test.generate_trading_signals(s))
    print("success")

if __name__ == "__main__":
    main()
