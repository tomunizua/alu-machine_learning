#!/usr/bin/env python3
"""
File defines a class that represents a deep neural network
"""


import numpy as np


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network
    """

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network
        with one hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        previous = nx
        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        Getter for the private instance attribute __L
        """
        return (self.__L)

    @property
    def cache(self):
        """
        Getter for the private instance attribute __cache
        """
        return (self.__cache)

    @property
    def weights(self):
        """
        Getter for the private instance attribute __weights
        """
        return (self.__weights)
