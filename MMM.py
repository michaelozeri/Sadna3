import json
from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np
import time

UPDATE_SIGNATURES_DATA = 0


class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):
        # defining the mmm
        self.log_signatures_data = self.convert_to_log_scale_eij(signatures_data)
        self.log_initial_pi = self.convert_to_log_scale(initial_pi)

        # constants - don't change
        self.dim_n = len(self.log_signatures_data)
        self.dim_m = len(self.log_signatures_data[0])
        self.dim_t = len(input_x)
        self.b_array = self.create_b_array(input_x, self.dim_m)

        # are calculated each iteration
        self.e_array = np.zeros((self.dim_n, self.dim_m))
        self.a_array = np.zeros(self.dim_n)

    def convert_to_log_scale_eij(self, signatures_data):
        return np.array(log(signatures_data))

    def convert_to_log_scale(self, initial_pi):
        return np.array(log(initial_pi))

    def create_b_array(self, input_x, m):
        b = np.zeros(m)
        for i in range(len(input_x)):
            b[int(input_x[i] - 1)] += 1
        return np.array(b)

    def log_to_regular(self, param):
        return exp(param)

    def fit(self, input_x_data, mmm_parameters):
        current_number_of_iterations = 1
        old_score = self.likelihood(input_x_data, mmm_parameters)
        self.e_step(mmm_parameters)
        self.m_step(mmm_parameters)
        new_score = self.likelihood(input_x_data, mmm_parameters)
        while (abs(new_score - old_score) > threshold) and (current_number_of_iterations < max_iteration):
            # print("delta is: " + abs(new_score - old_score).__str__())
            old_score = new_score
            self.e_step(mmm_parameters)
            # print(self.log_initial_pi)
            self.m_step(mmm_parameters)
            # print(self.log_initial_pi)
            new_score = self.likelihood(input_x_data, mmm_parameters)
            current_number_of_iterations += 1
            # print("number of iterations is: " + number_of_iterations.__str__())
        return

    def e_step(self, mmm_parameters):
        # this is the correct calc for the Eij by the PDF
        for i in range(self.dim_n):
            for j in range(self.dim_m):
                temp_log_sum_array = self.log_initial_pi + self.log_signatures_data[:, j]
                self.e_array[i][j] = (
                        log(self.b_array[j]) + self.log_initial_pi[i] + self.log_signatures_data[i][j] - logsumexp(
                    temp_log_sum_array))
        # this is from the mail with itay to calculate log(Ai)
        self.a_array = logsumexp(self.e_array, axis=1)

    # checks convergence from formula
    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(self, input_x_data, mmm_parameters):
        convergence = 0
        for t in range(self.dim_t):
            temp_log_sum_array = self.log_initial_pi + self.log_signatures_data[:,
                                                       int(input_x_data[int(t)])]
            convergence += logsumexp(temp_log_sum_array)
        return convergence

    def m_step(self, mmm_parameters):
        self.log_initial_pi = self.a_array - log(self.dim_t)
        if UPDATE_SIGNATURES_DATA:
            for i in range(self.dim_n):
                for j in range(self.dim_m):
                    # numerically stable for pi - Eij is already log(Eij)
                    self.log_signatures_data[i][j] = self.e_array[i][j] - log(
                        sum(self.log_to_regular(self.e_array), axis=1)[j])
