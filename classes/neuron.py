""" Neuron class module
"""

from typing import List, Tuple
from random import random
from copy import deepcopy
from interfaces.neuron_interface import NeuronInterface
from features_collections.weights_initialize import InitWeights
from classes.errors.exceptions import BadNeuronInitializationException, InvalidNeuronInputException


class Neuron(NeuronInterface):
    __weights = []
    __bias = None
    __neuron_length = 0

    def __init__(self,
                 inputs_number: int,
                 weights: List = None,
                 bias: float = None,
                 initial_format: InitWeights = InitWeights.ZEROS
        ) -> None:
        """
        A single neuron of the neural network.
        :param int inputs_number: the number of inputs of the neuron
        :param List weights: the initial weights list of neuron
        :param float bias: the initial bias value of neuron
        :param initial_format: the initialization format of weights
        :type initial_format: InitWeights
        """
        if weights is None:
            if inputs_number is None:
                raise BadNeuronInitializationException("At least the weights list or the number of inputs must be informed")
            self.__weights = [1 for i in range(inputs_number)] if initial_format == InitWeights.ONES \
                        else [0 for i in range(inputs_number)] if initial_format == InitWeights.ZEROS \
                        else [random() for i in range(inputs_number)] if initial_format == InitWeights.RANDOM \
                        else None
            if self.__weights is None:
                raise BadNeuronInitializationException("Invalid weights initialization format")
        elif len(weights) != inputs_number:
            raise BadNeuronInitializationException("The length of weights is not equal to the number of inputs")
        else:
            self.__weights = list(weights)
        
        if bias is None:
            bias = 1 if initial_format == InitWeights.ONES \
              else 0 if initial_format == InitWeights.ZEROS \
              else random() if initial_format == InitWeights.RANDOM \
              else None
            if bias is None:
                raise BadNeuronInitializationException("Invalid weights initialization format")
        else:
            self.__bias = bias / 1 # The division ensures that the bias is a numeric value
        
        self.__neuron_length = len(self.__weights)

    def calculate(self, input: Tuple) -> float:
        if len(input) != self.__neuron_length:
            raise InvalidNeuronInputException("The number of inputs is not valid")
        res = self.__bias
        for i in range(self.__neuron_length):
            res += self.__weights[i] * input[i]
        return res
    
    def actualize_weights(self, delta_weights: Tuple) -> None:
        if len(delta_weights) != self.__neuron_length:
            raise InvalidNeuronInputException("The number of inputs is not valid")
        for i in range(self.__neuron_length):
            self.__weights[i] += delta_weights[i]
    
    def get_weights(self) -> Tuple:
        return deepcopy(self.__weights)
    
    def actualize_bias(self, delta_bias: float) -> None:
        self.__bias += delta_bias
    
    def get_bias(self) -> float:
        return deepcopy(self.__bias)
