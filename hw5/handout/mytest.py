import unittest
import numpy as np
import pickle as pk
from numpy.testing import assert_allclose

from neuralnet import (
    Linear, Sigmoid, SoftMaxCrossEntropy, NN,
    zero_init, random_init
)

TOLERANCE = 1e-4

with open("unittest_data.pk", "rb") as f:
    data = pk.load(f)

with open("raw_tests", 'w') as file:
    file.write(str(data))
    file.close