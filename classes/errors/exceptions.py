""" Exceptions module
"""


from typing import Any


class AutogradDeactivatedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class BadNeuronInitializationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class InvalidNeuronInputException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
