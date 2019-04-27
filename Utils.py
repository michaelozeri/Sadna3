import json
import time
import numpy as np
from numpy import log, sum, amax, exp, shape


def create_random_array(array_size):
    random_array = np.random.rand(array_size)
    return np.array(random_array / random_array.sum(axis=0, keepdims=1))


def convert_to_log_scale_eij(signatures_data):
    return np.array(log(signatures_data))


def convert_to_log_scale(initial_pi):
    return np.array(log(initial_pi))


def create_b_array(input_x, m):
    b = np.zeros(m)
    for i in range(len(input_x)):
        b[int(input_x[i] - 1)] += 1
    return np.array(b)


def log_to_regular(param):
    return exp(param)
