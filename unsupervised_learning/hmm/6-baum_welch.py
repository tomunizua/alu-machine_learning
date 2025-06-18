#!/usr/bin/env python3
'''Performing the Baum-Welch algorithm for a hidden Markov model'''

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''Performs the forward algorithm for a hidden markov model'''
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


def backward(Observation, Emission, Transition, Initial):
    '''Performs the backward algorithm for a hidden markov model'''
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    '''Function that performs the Baum-Welch algorithm for a hidden Markov model'''
    N = Transition.shape[0]
    M = Emission.shape[1]
    T = Observations.shape[0]

    for _ in range(iterations):
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = (np.dot(np.dot(F[:, t].T, Transition) *
                                  Emission[:, Observations[t + 1]] *
                                  B[:, t + 1].T, 1))
            for i in range(N):
                numerator = (F[i, t] * Transition[i] *
                             Emission[:, Observations[t + 1]] *
                             B[:, t + 1].T)
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for i in range(M):
            Emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))

    return Transition, Emission
