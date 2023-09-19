""" Collections of weights initilaization module
"""

from enum import Enum


InitWeights = Enum("Initiate weights format",
                   [
                       'ZEROS',
                       'ONES',
                       'RANDOM'
                   ])