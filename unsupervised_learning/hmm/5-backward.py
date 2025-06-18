#!/usr/bin/env python3
'''Performing the backward algorithm for a hidden markov model'''

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''Function that performs the backward algorithm for a hidden markov model

    Parameters:
    Observation (numpy.ndarray): Index of the observation
    Emission (numpy.ndarray): Emission probability given a hidden state
    Transition (numpy.ndarray): Transition probabilities
    Initial (numpy.ndarray): Probability of starting in a hidden state

    Returns:
    P (float): Likelihood of the observations given the model
    B (numpy.ndarray): Backward path probabilities
    '''

    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    N1, N2 = Transition.shape
    if N1 != N or N2 != N:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    N3, N4 = Initial.shape
    if N3 != N or N4 != 1:
        return None, None
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for i in range(T - 2, -1, -1):
        B[:, i] = np.dot(Transition, B[:, i + 1] *
                         Emission[:, Observation[i + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
