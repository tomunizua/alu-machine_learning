#!/usr/bin/env python3
'''
This module deals with Bayesian Optimization
'''
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    '''
    Performs Bayesian optimization on a 1D Gaussian process.
    '''

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        '''
        Initializes Bayesian Optimization.
        '''
        MIN, MAX = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(MIN, MAX, num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        '''
        Calculates the next sample location using Expected Improvement.
        '''
        mu, _ = self.gp.predict(self.gp.X)
        sample_mu, sigma = self.gp.predict(self.X_s)

        opt_mu = np.min(mu) if self.minimize else np.max(mu)
        imp = opt_mu - sample_mu - self.xsi
        Z = imp / sigma
        EI = (imp * norm.cdf(Z)) + (sigma * norm.pdf(Z))
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, np.array(EI)

    def optimize(self, iterations=100):
        '''
        Optimizes the black-box function.
        '''
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break

            Y = self.f(X_next)
            self.gp.update(X_next, Y)

        idx = np.argmin(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = np.array(self.gp.Y[idx])
        return X_opt, Y_opt
