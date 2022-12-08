import pandas as pd
import numpy as np

from sklearn.decomposition import PCA, KernelPCA

class StatArbStrategy:
    def __int__(self):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010].
        :param
        :param
        :return:
        """
        pass

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
        # this is an edit
        return None
        # This is another test
    def estimate_resid_returns(self):
        """
        Estimate residual returns of token S.
        :return:
        """
        'HELLO TEST'
        pass

    def get_s_score(self):
        """
        Calculate s-score at time t.
        :return:
        """
        pass

    def deploy_regressor(self):
        """
        Apply linear regression and get parameters.
        :return:
        """

    def generate_trading_signals(self):
        """
        Generate trading signals at time t.
        :return:
        """
        pass

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