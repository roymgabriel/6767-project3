import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class Factors:
    def __init__(self):
        pass

    # Get the correlation matrix from the returns
    def get_corr_mat(self, df):
        return df.corr()

    # Get the PCA values for the given matrix of returns
    def get_pca(self, df):
        emp_corr = self.get_corr_mat(df)
        pca_model = PCA(n_components=2)
        pca_model.fit(emp_corr)
        eigenvalues = pca_model.explained_variance_
        eigenvectors = pd.DataFrame(pca_model.components_, columns=df.columns)
        # Returns the eigen vectors
        return eigenvectors, eigenvalues

    # Returns the Q matrix and the corresponding eigen vectors
    def get_Q(self, df):
        pca_eigenvectors, _ = self.get_pca(df)
        Q = pca_eigenvectors / df.std()
        return Q, pca_eigenvectors

    # Returns the factor returns for the given input
    def get_factor_return(self, ret_df, Q_j_df):
        # Q is 2x40 and hourly_rets 239x40
        return Q_j_df @ ret_df.T

    def get_eigen_port_returns(self):
        Q_j = self.Q_j.dropna(axis=1).fillna(0)
        rets = self.returns_df[Q_j.columns].loc[self.start].fillna(0)
        return pd.Series(np.nansum(Q_j.mul(rets), axis=1), name=self.start)

