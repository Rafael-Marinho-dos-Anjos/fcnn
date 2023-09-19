""" Sigmoid activation function module
"""

from interfaces.activation_function_interface import ActivationFunctionInterface
from math import exp, log


class Sigmoid(ActivationFunctionInterface):
    __auto_grad = False
    __last_in = None
    __last_out = None

    def __init__(self) -> None:
        """
        Sigmoid activation function.
        """
        super().__init__()
    
    def calculate(self, input: float) -> float:
        res = 1 / (1 + exp((-1) * input))
        if self.__auto_grad:
            self.__last_in = input
            self.__last_out = res
        return res

    def derivate(input: float) -> float:
        res = log(exp((-1) * input) + 1) + input
        return res
