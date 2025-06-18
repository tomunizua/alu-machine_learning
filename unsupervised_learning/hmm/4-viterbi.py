#!/usr/bin/env python3
'''This module Calculates the most likely sequence of
hidden states for a hidden markov model'''

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    '''Calculates the most likely sequence of hidden states

    Parameters:
    Observation (numpy.ndarray): Index of the observation
    Emission (numpy.ndarray): Emission probability given a hidden state
    Transition (numpy.ndarray): Transition probabilities
    Initial (numpy.ndarray): Probability of starting in a hidden state

    Returns:
    path (list): Most likely sequence of hidden states
    P (float): Probability of obtaining the path sequence
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
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    back = np.zeros((N, T))
    for i in range(1, T):
        F[:, i] = np.max(
            F[:, i - 1] * Transition.T * Emission[np.newaxis, :,
                                                  Observation[i]].T, axis=1)
        back[:, i] = np.argmax(
            F[:, i - 1] * Transition.T, axis=1)
    P = np.max(F[:, -1])
    Path = [np.argmax(F[:, -1])]
    for i in range(T - 1, 0, -1):
        Path.insert(0, int(back[Path[0], i]))
    return Path, P
