""" Activation function interface module
"""

from abc import ABC, abstractmethod
from typing import List
from classes.errors.exceptions import AutogradDeactivatedException


class ActivationFunctionInterface(ABC):
    __auto_grad = False
    __last_out = None
    __learning_rate = 0.001

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

    def backpropagate(self,
                      input: float,
                      desired_out: float
        ) -> float:
        """
        Automaticly calculate the backpropagation value for a neuron
        weight with the stored gradient values.
        :param float input: input value for the weight of the neuron
        :param float desired_out: desired neuron output for this input
        :return: float
        """
        if not self.__auto_grad:
            raise AutogradDeactivatedException("You must activate the autograd before use this feature.")
        drvt = self.derivate(self.__last_out)
        loss = desired_out - self.__last_out
        delta_weight = (-1) * self.__learning_rate * drvt * input * loss
        return delta_weight

    def auto_grad(self, activate: bool = True) -> None:
        """
        Activate or deactivate the gradient auto storage (used for
        a faster backpropagation).
        :param bool activate: activates autograd if True (deactivates if False)
        """
        self.__auto_grad = True if activate else False

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Sets the learning rate applied to backpropagation.
        :param float learning_rate: new learning rate value
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be a positive number.")
        self.__learning_rate = learning_rate
