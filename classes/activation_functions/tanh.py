""" Hyperbolic tangent activation function module
"""

from interfaces.activation_function_interface import ActivationFunctionInterface
from math import tanh


class Tanh(ActivationFunctionInterface):
    __auto_grad = False
    __last_in = None
    __last_out = None

    def __init__(self) -> None:
        """
        Hyperbolic tangent activation function.
        """
        super().__init__()
    
    def calculate(self, input: float) -> float:
        res = tanh(input)
        if self.__auto_grad:
            self.__last_in = input
            self.__last_out = res
        return res

    def derivate(input: float) -> float:
        res = 1 + tanh(input)**2
        return res
