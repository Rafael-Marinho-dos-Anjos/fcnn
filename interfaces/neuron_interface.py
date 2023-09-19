""" Neuron interface module
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class NeuronInterface(ABC):
    @abstractmethod
    def calculate(self, input: Tuple) -> float:
        """
        Calculates the neuron output.
        :param list input: the input values of the neuron
        :return: float
        """
    
    @abstractmethod
    def actualize_weights(self, delta_weights: Tuple) -> None:
        """
        Actualizates the weights of the neuron.
        :param tuple delta_weights: the actualization value for each neuron weight
        """

    @abstractmethod
    def get_weights(self) -> Tuple:
        """
        Returns a list with all neuron weights.
        :return: Tuple
        """

    @abstractmethod
    def actualize_bias(self, delta_bias: float) -> None:
        """
        Actualizates the bias value of the neuron.
        :param float delta_bias: the actualization value of bias
        """

    @abstractmethod
    def get_bias(self) -> float:
        """
        Returns the current bias value.
        :return: float
        """
