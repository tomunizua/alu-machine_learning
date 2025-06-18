#!/usr/bin/env python3
'''This module performs the forward algorithm for a hidden markov model'''

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''Performs the forward algorithm for a hidden markov model

    Parameters:
    Observation (numpy.ndarray): Index of the observation
    Emission (numpy.ndarray): Emission probability given a hidden state
    Transition (numpy.ndarray): Transition probabilities
    Initial (numpy.ndarray): Probability of starting in a hidden state

    Returns:
    P (float): Likelihood of the observations given the model
    F (numpy.ndarray): Forward path probabilities
    '''

    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    N1, N2 = Transition.shape
    if N1 != N or N2 != N:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    N3, N4 = Initial.shape
    if N3 != N or N4 != 1:
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        F[:, i] = np.sum(
            F[:, i - 1] * Transition.T *
            Emission[np.newaxis, :, Observation[i]].T, axis=1)
    P = np.sum(F[:, -1])
    return P, F
