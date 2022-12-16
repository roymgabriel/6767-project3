import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from Factors import Factors


class StatArbStrategy:
    def __init__(self, window, start, finish, symbols_df, st_ret_df):
        """
        A class that implements a statistical arbitrage strategy as proposed in [Avellaneda and Lee 2010].
        :param window: the time window-size
        :param start: the start datetime
        :param finish: the finish datetime
        :param symbols_df: the coins' symbols csv filename
        :param st_ret_df: the returns csv filename
        """
        # Initialize variables
        self.window = window
        self.start = start
        self.finish = finish
        self.symbols_df = symbols_df
        self.st_ret_df = st_ret_df

        # creating factors object to get returns of the data and use the functions to calculate PCA
        self.factors = Factors()


    def get_data_window(self, t, win):
        """
        A function that creates the window needed to run the strategy
        :param t: datetime
        :param win: window
        :return: dataframe at desired window
        """
        used_symbols = list(self.symbols_df.loc[t])
        time_idx = self.st_ret_df.index.get_loc(t)
        win_df = self.st_ret_df[used_symbols].iloc[time_idx - win: time_idx]
        win_df = win_df.loc[:, (win_df != 0).any(axis=0)]
        return win_df

    def estimate_resid_returns(self, factor_returns, hourly_returns):
        """
        Estimate residual returns of token S.
        :param factor_returns: factor returns
        :param hourly_returns: hourly returns
        :return: the residuals from the regression model
        """

        regressor = LinearRegression()
        X = factor_returns  # 2x40
        y = hourly_returns  # 239x40
        d = {}
        params = {}
        for col in y.columns:
            model = regressor.fit(X.T, y[col])
            y_pred = model.predict(X.T)
            resid = y[col] - y_pred
            d[col] = resid
            params[col] = [model.intercept_] + model.coef_.tolist()  # b_0, b_1, b_2
        return pd.DataFrame(d)

    def get_X_l(self, residuals):
        """
        Function to get X_l in order to obtain a and b parameters
        :param residuals: the residuals from the above regression model
        :return: the a and b parameters from the autoregressive model
        """
        # NOTE: this function should return parameters a and b for each asset
        X_l = residuals.cumsum()
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
        return pd.DataFrame.from_dict(coeffs, orient='index', columns=['b', 'a']).T, pd.DataFrame.from_dict(
            xl_residuals, orient='columns')

    def get_params(self, X_l, X_l_residuals):
        """
        A function to get greek parameters
        :param X_l: The X value from the autoregressive model
        :param X_l_residuals: the residuals from the autoregressive model (zeta)
        :return:the greek parameters in a dataframe
        """
        # I changed the function to loop through each currency and get the parameters for each one
        params = {}
        for col in X_l.columns:
            # Defining the input parameters for each currency
            a = X_l.loc['a', col]
            b = X_l.loc['b', col]
            resid = X_l_residuals[col]

            kappa = -np.log(b) * 8760  # For the number of hours in a year
            m = a / (1 - b)
            sigma = np.sqrt(np.var(resid) * 2 * kappa / (1 - b ** 2))
            sigma_eq = np.sqrt(np.var(resid) / (1 - b ** 2))
            params[col] = [kappa, m, sigma, sigma_eq]
        return pd.DataFrame.from_dict(params, orient='index', columns=['kappa', 'm', 'sigma', 'sigma_eq'])

    def get_s_score(self, params_df):
        """
        Calculate s-score at time t.
        :param params_df: the parameter dataframe
        :return: s_score value
        """
        m_mean = params_df['m'].mean()
        s_score = (m_mean - params_df['m']) / params_df['sigma_eq']
        return s_score

    def generate_trading_signals(self, s_score, s_bo=1.25, s_so=1.25, s_bc=0.75, s_sc=0.5):
        """
        A function that generates the trading signals across the entire time period.
        :param s_score: s_score value
        :param s_bo: BTO trading signal
        :param s_so: STO trading signal
        :param s_bc: BTC trading signal
        :param s_sc: STC trading signal
        :return: the trading signals in a dataframe format
        """
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

    # Module of the arbitrage strategy that computes the s_scores and signals of the given window based on the top 40 currencies
    # Also returning the additional parameters needed for the project tasks
    def get_signal(self, t, window):
        """
        A module of the arbitrage strategy that computes the s_scores and signals of the given window
         based on the top 40 currencies. It also returns the additional parameters needed for the project tasks.
        :param t: datetime
        :param window: window-size
        :return: s_score, signals, Q, eigen_vects
        """
        st_ret = self.get_data_window(t, window)
        Q, eigen_vects= self.factors.get_Q(st_ret)
        factor_rets = self.factors.get_factor_return(st_ret, Q).replace([np.inf, -np.inf], np.nan).fillna(0)
        residuals = self.estimate_resid_returns(factor_rets, st_ret).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_l, X_l_residuals = self.get_X_l(residuals)
        params = self.get_params(X_l.replace([np.inf, -np.inf], np.nan).fillna(0), X_l_residuals.replace([np.inf, -np.inf], np.nan).fillna(0))
        s_score = self.get_s_score(params)
        signals = self.generate_trading_signals(s_score)
        return s_score, signals, Q, eigen_vects


    def get_trading_signals(self, start, finish, window):
        """
        A function the trades on the signals generated after iterating through every hour between start
        and end datetime.
        :param start: start datetime
        :param finish: end datetime
        :param window: window-size
        :return: a dictionary of all the results
        """
        date_range = pd.date_range(start, finish, freq='H')

        cols = []
        trading_signals = []
        egn_portfolio_wgts1 = []
        egn_portfolio_wgts2 = []
        egn_vect1 = []
        egn_vect2 = []
        s_score_list = []
        egn_portfolios_ret = []
        for start_time in date_range:
            print(start_time)
            s_score, signals, Q, eigen_vects = self.get_signal(start_time, window)

            # Creating the return dataframe
            cols.append(Q.columns)
            egn_portfolio_wgts1.append(Q.iloc[0, :].rename(start_time))
            egn_vect1.append(eigen_vects.iloc[0, :].rename(start_time))
            egn_portfolio_wgts2.append(Q.iloc[1, :].rename(start_time))
            egn_vect2.append(eigen_vects.iloc[1, :].rename(start_time))
            trading_signals.append(signals.rename(start_time))
            s_score_list.append(s_score.rename(start_time))

        cols = np.unique(Q.columns)
        sig_df = pd.DataFrame(trading_signals, columns=cols)
        eport_df1 = pd.DataFrame(egn_portfolio_wgts1, columns=cols)
        evect_df1 = pd.DataFrame(egn_vect1, columns=cols)
        eport_df2 = pd.DataFrame(egn_portfolio_wgts2, columns=cols)
        evect_df2 = pd.DataFrame(egn_vect2, columns=cols)
        s_score_df = pd.DataFrame(s_score_list, columns=cols)
        results_list = [sig_df, eport_df1, evect_df1, eport_df2, evect_df2, s_score_df]
        results_name_list = ['signals', 'eigportwgts1', 'eigportwgts2', 'eigvect1', 'eigvect2', 'score']
        return dict(zip(results_name_list, results_list))


