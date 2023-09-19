""" Activation functions module
"""

from interfaces.activation_function_interface import ActivationFunctionInterface


class ReLu(ActivationFunctionInterface):
    __auto_grad = False
    __last_in = None
    __last_out = None

    def calculate(self, input: float) -> float:
        res = input if input > 0 else 0
        if self.__auto_grad:
            self.__last_in = input
            self.__last_out = res
        return res

    def derivate(input: float) -> float:
        res = 1 if input > 0 else 0
        return res
    
    def backpropagate(self) -> float:
        if not self.__auto_grad:
            # TODO -> Criar um erro para autograd desativado.
            raise TypeError("")
        drvt = self.derivate(self.__last_in)
        # TODO -> Receber os valores de erro para atualizar os
        #         pesos do neurônio.

    def auto_grad(self, activate: bool = True) -> None:
        self.__auto_grad = True if activate else False