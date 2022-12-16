import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class Factors:
    def __init__(self):
        """
        A class that initializes the eigenportfolio weights, vectors, and returns needed to
        run the Statistical Arbitrage Strategy.
        """
        pass

    def get_corr_mat(self, df):
        """
        A function that computes the correlation matrix from the returns.
        :param df: return data
        :return: correlation matrix
        """
        return df.corr().fillna(0)

    # Get the PCA values for the given matrix of returns
    def get_pca(self, df):
        """
        A functoin that computes the PCA values based on a given empirical return matrix
        :param df: return data
        :return: eigenvectors, eigenvalues
        """
        emp_corr = self.get_corr_mat(df)
        pca_model = PCA(n_components=2)
        pca_model.fit(emp_corr)
        eigenvalues = pca_model.explained_variance_
        eigenvectors = pd.DataFrame(pca_model.components_, columns=df.columns)
        # Returns the eigen vectors
        return eigenvectors, eigenvalues

    def get_Q(self, df):
        """
        A function that returns the eigenportfolios (Q_matrix) and their corresponding eigenvalues.
        :param df: return data
        :return: Q, pca_eigenvectors
        """
        pca_eigenvectors, _ = self.get_pca(df)
        Q = pca_eigenvectors / df.std()
        return Q, pca_eigenvectors

    # Returns the factor returns for the given input
    def get_factor_return(self, ret_df, Q_j_df):
        """
        A function that computes the factor returns for a given input of Q and a return series.
        :param ret_df: The return series
        :param Q_j_df: The eigenportfolios
        :return: Factor returns F
        """
        # Q is 2x40 and hourly_rets 239x40
        return Q_j_df @ ret_df.T

    def get_eigen_port_returns(self):
        """
        A function that computes the eigenportfolio returns
        :return: eigenportfolio returns
        """
        Q_j = self.Q_j.dropna(axis=1).fillna(0)
        rets = self.returns_df[Q_j.columns].loc[self.start].fillna(0)
        return pd.Series(np.nansum(Q_j.mul(rets), axis=1), name=self.start)

