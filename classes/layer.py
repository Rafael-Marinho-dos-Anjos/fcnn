""" Neural network layer module
"""

from typing import Tuple
from interfaces.layer_interface import LayerInterface, ActivationFunctionInterface
from classes.neuron import Neuron, InitWeights, deepcopy


class Layer(LayerInterface):
    def __init__(self,
                 inputs_number: int,
                 outputs_number: int,
                 activation_function: ActivationFunctionInterface,
                 initiation_weights: InitWeights) -> None:
        self.__neurons = [Neuron(inputs_number=inputs_number,
                                 initial_format=initiation_weights)
                                 for i in outputs_number]
        activation_function = activation_function if isinstance(activation_function, ActivationFunctionInterface) \
                              else activation_function()
        self.__activations = [deepcopy(activation_function) for i in outputs_number]

    def foward(self, input: Tuple) -> Tuple:
        output = [neuron.calculate(input) for neuron in self.__neurons]
        self.__last_inputs = input
        return tuple(output)

    def backpropagate(self, expected_out: Tuple) -> None:
        for index, neuron in enumerate(self.__neurons):
            actualizations = [self.__activations.backpropagate(input_, expected_out[index]) for input_ in self.__last_inputs]
            neuron.actualize_weights(actualizations)
            bias_actualization = self.__activations.backpropagate(1, expected_out[index])
            neuron.actualize_bias(bias_actualization)

    def autograd(self, activate: bool = True) -> None:
        for function in self.__activations:
            function.auto_grad(activate)
