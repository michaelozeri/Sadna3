import json
from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np
import time
import Utils

UPDATE_SIGNATURES_DATA = 0


class MMM:
    def __init__(self, signatures_data, initial_pi, input_x):
        # defining the mmm
        self.log_signatures_data = Utils.convert_to_log_scale_eij(signatures_data)
        self.log_initial_pi = Utils.convert_to_log_scale(initial_pi)

        # constants - don't change
        self.dim_k = len(self.log_signatures_data)
        self.dim_m = len(self.log_signatures_data[0])
        self.dim_t = len(input_x)
        self.b_array = Utils.create_b_array(input_x, self.dim_m)

        # are calculated each iteration
        self.e_array = np.zeros((self.dim_k, self.dim_m))
        self.a_array = np.zeros(self.dim_k)

    def fit(self, input_x_data, threshold, max_iteration):
        current_number_of_iterations = 1
        old_score = self.likelihood(input_x_data)
        self.e_step()
        self.m_step()
        new_score = self.likelihood(input_x_data)
        while (abs(new_score - old_score) > threshold) and (current_number_of_iterations < max_iteration):
            old_score = new_score
            self.e_step()
            self.m_step()
            new_score = self.likelihood(input_x_data)
            current_number_of_iterations += 1
        return

    def e_step(self):
        # this is the correct calc for the Eij by the PDF
        k_array = [logsumexp((self.log_initial_pi + self.log_signatures_data[:, j])) for j in range(self.dim_m)]
        self.e_array = [(log(self.b_array) + self.log_initial_pi[i] + self.log_signatures_data[i] - k_array) for i in range(self.dim_k)]
        # this is from the mail with itay to calculate log(Ai)
        self.a_array = logsumexp(self.e_array, axis=1)

    # checks convergence from formula
    # on input on input data (sequence or sequences), return log probability to see it
    def likelihood(self, input_x_data):
        return np.sum([logsumexp(self.log_initial_pi + self.log_signatures_data[:, int(input_x_data[t])]) for t in
                       range(self.dim_t)])

    def m_step(self):
        self.log_initial_pi = self.a_array - log(self.dim_t)
        if UPDATE_SIGNATURES_DATA:
            for i in range(self.dim_k):
                for j in range(self.dim_m):
                    # numerically stable for pi - Eij is already log(Eij)
                    self.log_signatures_data[i][j] = self.e_array[i][j] - log(
                        sum(Utils.log_to_regular(self.e_array), axis=1)[j])
