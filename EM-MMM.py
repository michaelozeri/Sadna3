import json
from numpy import log, sum, amax, exp, shape
from scipy.special import logsumexp
import numpy as np
import time

# we don't want to update signatures array (itay asked) at this point so i made
# a global to set if to update the signatures data or not at this time
LOG_INITIAL_PI_KEY = "log_initial_pi"
LOG_SIGNATURES_DATA_KEY = "log_signatures_data"
A_ARRAY_KEY = "a_array"
DIM_N_KEY = "dim_n"
E_ARRAY_KEY = "e_array"
B_ARRAY_KEY = "b_array"
DIM_M_KEY = "dim_m"
DIM_T_KEY = "dim_t"
UPDATE_SIGNATURES_DATA = False

######### CROSS VALIDATION FIELDS #######

threshold = 0.01
max_iteration = 1000


############################################## CROSS VALIDATION FUNCTIONS ##############################################


def compute_likelihood_for_chromosome(ignored_chromosome, person, initial_pi, signatures_data):
    input_x_total = np.array([])
    # train
    for chromosome in person:
        if chromosome == ignored_chromosome:
            continue
        else:
            input_x_total = np.append(input_x_total, np.array(person[chromosome]["Sequence"]))
            # input_x_total.extend(person[chromosome]["Sequence"])
    mmm_parameters = initialize_mmm_parameters(signatures_data, initial_pi, input_x_total)
    fit(input_x_total, mmm_parameters)
    ignored_sequence = person[ignored_chromosome]["Sequence"]
    mmm_parameters[DIM_T_KEY] = len(ignored_sequence)
    mmm_parameters[B_ARRAY_KEY] = create_b_array(ignored_sequence, mmm_parameters[DIM_M_KEY])
    # set_t(len(ignored_sequence))
    # set_b(ignored_sequence)
    return likelihood(ignored_sequence, mmm_parameters)


def person_cross_validation(person, initial_pi, signatures_data):
    total_sum_person = 0
    for ignored_chromosome in person:
        total_sum_person += compute_likelihood_for_chromosome(ignored_chromosome, person, initial_pi, signatures_data)
        print("total_sum_person is: " + str(total_sum_person) + " after running chromosome: " + str(ignored_chromosome))
    return total_sum_person


def compute_cross_validation_for_total_training_data(dict_data, initial_pi, signatures_data):
    total_sum = 0
    person_number = 1
    for person in dict_data:
        start = time.time()
        total_sum += person_cross_validation(dict_data[person], initial_pi, signatures_data)
        print("total sum for person: " + str(person_number) + " is: " + str(total_sum))
        end = time.time()
        print("Execution time for person " + str(person_number) + " is: " + str(end - start) + " Seconds, " + str(
            (end - start) / 60) + " Minutes.")
        person_number += 1
    return total_sum


############################################## START RUN OF FILE ##############################################


def main_single_fit():
    # read example data from JSON
    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = (data['initial_pi'])
    trained_pi = data['trained_pi']
    input_x = data['input']

    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open('data/ICGC-BRCA.json') as f1:
        dic_data = json.load(f1)

    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.load("data/BRCA-signatures.npy")

    print("started the init")

    mmm_parameters = initialize_mmm_parameters(signatures_data, initial_pi, input_x)

    fit(input_x, mmm_parameters)

    err = 0
    for i in range(len(initial_pi)):
        err += abs(log_to_regular(mmm_parameters[LOG_INITIAL_PI_KEY][i]) - trained_pi[i])
        # print(abs(mmm.log_to_regular(mmm.log_initial_pi[i]) - trained_pi[i]))

    print(err)
    # print(mmm.likelihood(dic_data))


def main_algorithm_1():
    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open('data/ICGC-BRCA.json') as f1:
        dic_data = json.load(f1)

    with open('data/example.json') as f:
        data = json.load(f)
    initial_pi = np.array(data['initial_pi'])

    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = np.array(np.load("data/BRCA-signatures.npy"))

    print("Started cross validation for 1'st type algo")

    training = compute_cross_validation_for_total_training_data(dic_data, initial_pi, signatures_data)
    print("Total sum is: " + str(training))


# function call
main_algorithm_1()
