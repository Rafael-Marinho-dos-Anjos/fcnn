""" ReLu activation function module
"""

from interfaces.activation_function_interface import ActivationFunctionInterface


class ReLu(ActivationFunctionInterface):
    def __init__(self) -> None:
        """
        ReLu activation function.
        """
        super().__init__()

    def calculate(self, input: float) -> float:
        res = input if input > 0 else 0
        if self.__auto_grad:
            self.__last_in = input
            self.__last_out = res
        return res

    def derivate(input: float) -> float:
        res = 1 if input > 0 else 0
        return res
