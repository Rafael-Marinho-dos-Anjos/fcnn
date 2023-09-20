""" Neural network layer interface module
"""

from abc import ABC, abstractmethod
from typing import Tuple
from interfaces.activation_function_interface import ActivationFunctionInterface
from features_collections.weights_initialize import InitWeights


class LayerInterface(ABC):
    __neurons = []
    __activations = []
    __last_inputs = []

    @abstractmethod
    def __init__(self,
                 inputs_number: int,
                 outputs_number: int,
                 activation_function: ActivationFunctionInterface,
                 initiation_weights: InitWeights
        ) -> None:
        """
        A single layer of the neural network.
        :param int inputs_number: the number of inputs of the layer
        :param int outputs_number: the number of outputs of the layer
        :param activation_function: the activation function used on layer
        :type activation_function: ActivationFunctionInterface
        :param initiation_weights: the type of initialization of the weights values
        :type initiation_weights: InitWeights
        """

    @abstractmethod
    def foward(self, input: Tuple) -> Tuple:
        """
        Calculates the foward pass of the layer.
        :param tuple input: the imput of layer
        :return: tuple
        """

    @abstractmethod
    def backpropagate(self, expected_out: Tuple) -> None:
        """
        Aplies the backpropagation to actualize the weights on layer neurons.
        :param tuple expected_out: the expected output of layer
        """

    @abstractmethod
    def autograd(self, activate: bool = True) -> None:
        """
        Activate or deactivate the gradient auto storage (used for
        a faster backpropagation).
        :param bool activate: activates autograd if True (deactivates if False)
        """
