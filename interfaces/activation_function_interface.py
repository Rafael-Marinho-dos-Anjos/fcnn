""" Activation function interface module
"""

from abc import ABC, abstractmethod
from typing import List


class ActivationFunctionInterface(ABC):
    @abstractmethod
    def calculate(self, input: float) -> float:
        """
        Calculates the value of the neuron output after
        passing by the activation function.
        :param float input: input of the Activation function (neuron output)
        :return: float
        """
    
    @abstractmethod
    def derivate(input: float) -> float:
        """
        Calculates the derivate value of function at the given
        point (used for backpropagation).
        :param float input: derivation point
        :return: float
        """

    @abstractmethod
    def backpropagate(self) -> float:
        """
        Automaticly calculate the backpropagation value for neuron
        with the stored gradient values.
        :return: float
        """

    @abstractmethod
    def auto_grad(self, activate: bool = True) -> None:
        """
        Activate or deactivate the gradient auto storage (used for
        a faster backpropagation).
        :param bool activate: activates autograd if True (deactivates if False)
        """
