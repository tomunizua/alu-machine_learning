#!/usr/bin/env python3
'''
1. Gaussian Process Prediction
'''
import numpy as np

class GaussianProcess:
    '''
    Represents a noiseless 1D Gaussian process.
    '''

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        '''
        Initializes the Gaussian Process.

        Args:
            X_init (np.ndarray): Inputs sampled from the function.
            Y_init (np.ndarray): Outputs of the function.
            l (float): Length scale for the kernel.
            sigma_f (float): Standard deviation of the output.
        '''
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X1=X_init, X2=X_init)

    def kernel(self, X1, X2):
        '''
        Computes the covariance kernel matrix.

        Args:
            X1 (np.ndarray): First input matrix.
            X2 (np.ndarray): Second input matrix.

        Returns:
            np.ndarray: Covariance matrix.
        '''
        cov = np.exp(-((X1 - X2.T) ** 2) / (2 * (self.l ** 2)))
        return (self.sigma_f ** 2) * cov

    def predict(self, X_s):
        '''
        Predicts mean and standard deviation for input points.

        Args:
            X_s (np.ndarray): Points for prediction.

        Returns:
            tuple: Predicted means and variances.
        '''
        s = X_s.size
        cov = self.kernel(self.X, X_s)
        solution = np.linalg.solve(self.K, cov).T
        cov2 = self.kernel(X_s, X_s)
        mu = solution @ self.Y
        sigma = cov2 - (solution @ cov)

        return mu.reshape(s,), np.diag(sigma)
