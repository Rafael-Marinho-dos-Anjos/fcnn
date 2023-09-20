""" Neural network model class module
"""

from typing import Tuple
from interfaces.model_interface import ModelInterface
from classes.errors.exceptions import BadModelInitialization
from classes.activation_functions.relu import ReLu
from classes.layer import Layer, InitWeights, deepcopy


class Model(ModelInterface):
    def __init__(self, *args, **kwargs) -> None:
        if "load_model" in kwargs:
            # TODO - fazer um load a partir de um modelo salvo
            return

        if len(args) < 2:
            raise BadModelInitialization("At least an input legnth and an output length must be provided")

        try:
            activation_func = kwargs["activation_function"]
        except:
            activation_func = ReLu()

        try:
            initial_weights = kwargs["initiation_weights"]
        except:
            initial_weights = InitWeights.ONES

        for i in range(len(args) - 1):
            self.__layers.append(Layer(args[i], args[i + 1], activation_func, initial_weights))

    def export_to_json(self, path: str) -> None:
        # TODO - fazer um savestate para um JSON
        pass

    def predict(self, input: Tuple) -> Tuple:
        out = deepcopy(input)
        for layer in self.__layers:
            out = layer.foward(out)
        self.__last_out = out
        return out

    def autograd(self, activate: bool) -> None:
        for layer in self.__layers:
            layer.autograd(activate)

    def backpropagate(self, expected_out: Tuple) -> None:
        # TODO - backpropagation
        pass
