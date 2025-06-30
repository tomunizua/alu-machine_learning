#!/usr/bin/env python3
'''
This module defines Bayesian Optimization(Acquisition)
'''
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess

class BayesianOptimization:
    '''
    Performs Bayesian optimization on a Gaussian process.
    '''

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
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

        if self.minimize:
            opt_mu = np.min(mu)
        else:
            opt_mu = np.max(mu)

        imp = opt_mu - sample_mu - self.xsi
        Z = imp / sigma
        EI = ((imp * norm.cdf(Z)) + (sigma * norm.pdf(Z)))
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, np.array(EI)
