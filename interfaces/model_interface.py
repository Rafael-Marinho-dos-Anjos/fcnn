""" Neural network model interface module
"""

from abc import ABC, abstractmethod
from typing import Tuple


class ModelInterface(ABC):
    __layers = []
    __last_out = []

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Fully connected neural network.
        :param int args: number of inputs on each layer (the last is the output length)
        :param activation_function: the used activation function (standart ReLu)
        :type activation_function: ActivationFunctionInterface
        :param initiation_weights: the initial value of the weights (standart ones)
        :type initiation_weights: InitWeights
        :param str load_model: loads trained model from a JSON archive
        """
    
    @abstractmethod
    def export_to_json(self, path: str) -> None:
        """
        Exports model to a JSON archive.
        :param str path: path to where the archive will be saved
        """

    @abstractmethod
    def predict(self, input: Tuple) -> Tuple:
        """
        Predict the model output from a given input.
        :param tuple input: a tuple with inputs to the model
        :return: tuple
        """

    @abstractmethod
    def autograd(self, activate: bool) -> None:
        """
        Activate or deactivate the gradient auto storage (used for
        a faster backpropagation).
        :param bool activate: activates autograd if True (deactivates if False)
        """
    
    @abstractmethod
    def backpropagate(self, expected_out: Tuple) -> None:
        """
        Aplies the backpropagation to actualize the weights on all layer neurons.
        :param tuple expected_out: the expected output of model
        """