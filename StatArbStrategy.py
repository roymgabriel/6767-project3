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

        self.residuals = self.estimate_resid_returns()
        self.X_l = self.get_X_l()

    def get_factor_returns(self):
        """
        Compute factor returns of the two risk factors at time t.
        :return:
        """

        pass

    def get_PCA(self):
        """
        Apply Principal Component Anlysis (PCA) and get top two pc vectors along with their eigenvalues.
        :return:
        """
        return None

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
        ''' FINISHED '''
        # TODO: We need to get the residuals of the future time (t + dt) as well and do regression on those
        # NOTE: this function should return parameters a and b for each asset
        X_l = self.residuals.expanding().sum()
        X_l_shift = X_l.shift(-1)
        regressor = LinearRegression()
        X = X_l[:-1] # M-1 rows
        y = X_l_shift[:-1] # M-1 rows
        d = {}
        for col in y.columns:
            model = regressor.fit(X[[col]], y[col])
            # Returns the a and b coefficients for each currency
            d[col] = [model.coef_[0], model.intercept_]
        return pd.DataFrame.from_dict(d, orient='index', columns=['b', 'a'])



    def get_params(self, a, b, resid):
        """
        A function to get greek parameters
        :return:
        """
        #TODO: we can either loop here or in init but basically we loop through each asset and get their
        # a and b and resid from self.xxx, then you do  the calculations below
        # we can store the results in a dictionary for each asset for later functions
        kappa = -np.log(b) * 252
        m = a / (1-b)
        sigma = np.sqrt(np.var(resid) * 2 * kappa / (1 - b**2))
        sigma_eq = np.sqrt(np.var(resid) / (1 - b**2))
        return kappa, m, sigma, sigma_eq
    def get_s_score(self, X_i, m_i, sigma_eq_i):
        """
        Calculate s-score at time t.
        :return:
        """
        # TODO: We need to do this for each asset, X_i is most confusing but it is basically X_l
        return (X_i - m_i) / sigma_eq_i

    def deploy_regressor(self):
        """
        Apply linear regression and get parameters.
        :return:
        """
        pass

    def generate_trading_signals(self, s_score):
        """
        Generate trading signals at time t.
        :return:
        """
        # TODO: We need to somehow translate those strings into actually going long or short these stocks, where do the
        # weights come in? maybe Q changes idk
        s_bo = 1.25
        s_so = 1.25
        s_bc = 0.75
        s_sc = 0.5

        if s_score < -s_bo:
            return "BTO"
        elif s_score > s_so:
            return "STO"
        elif s_score < s_bc:
            return "CSP"
        elif s_score > -s_sc:
            return "CLP"



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

    def get_SR(self):
        """
        Compute Sharpe Ratio.
        :return:
        """
        pass

    def get_MDD(self):
        """
        Compute maximum draw-down (MDD).
        :return:
        """
        pass

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
    print(test.get_X_l())
    print("success")

if __name__ == "__main__":
    main()
