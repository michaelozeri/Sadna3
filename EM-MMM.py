import json
import numpy as np
from MMM import MMM
import Utils
from scipy.special import logsumexp


######### CROSS VALIDATION FIELDS #######
threshold = 0.01
max_iteration = 1000
CHROMOSOME_NUMBER = 23
NUMBER_OF_PEOPLE = 560
DIM_K = 12
DIM_M = 96


############################################## CROSS VALIDATION FUNCTIONS ##############################################

def e_step_for_ignored_chromosome(ignored_chromosome, person, initial_pi, signatures_data):
    input_x_total = np.array([])
    # train
    for chromosome in person:
        if chromosome == ignored_chromosome:
            continue
        else:
            input_x_total = np.append(input_x_total, np.array(person[chromosome]["Sequence"]))
    mmm = MMM(signatures_data, initial_pi, input_x_total)
    mmm.e_step()
    return mmm


def build_input_x_on_other_chromosome_and_e_step(person, initial_pi, signatures_data, ignored_chromosome_number):
    ignored_chromosome = person[ignored_chromosome_number]  # TODO: check this
    mmm = e_step_for_ignored_chromosome(ignored_chromosome, person, initial_pi, signatures_data)
    return mmm, ignored_chromosome


def sum_all_e_arrays(all_person_mmm_array):
    pass


def compute_cross_validation_for_total_training_data(dict_data, initial_pi, signatures_data):
    cross_val_mat = np.array([NUMBER_OF_PEOPLE, CHROMOSOME_NUMBER])
    for chromosome_number in range(1, CHROMOSOME_NUMBER):
        all_person_mmm_array = []
        temp_ignored_chromosome = None
        for person in dict_data:
            mmm, temp_ignored_chromosome = build_input_x_on_other_chromosome_and_e_step(dict_data[person],
                                                                                        initial_pi,
                                                                                        signatures_data,
                                                                                        chromosome_number)
            all_person_mmm_array[chromosome_number] = mmm
        total_e = sum_all_e_arrays(all_person_mmm_array) #TODO:finish this shit
        person_number = 0
        for mmm in all_person_mmm_array:
            mmm.e_array = total_e
            mmm.m_step()
            cross_val_mat[chromosome_number][person_number] = mmm.likelihood(temp_ignored_chromosome) #TODO: verify temp_ignored is the input_X ok
        return cross_val_mat


############################################## START RUN OF FILE ##############################################

def main_algorithm_2_for_1_strand():
    # read dictionary data from JSON
    # each key is a persons data - and inside there is chromosomes 1-22,X.Y and their input x1,...xt
    with open('data/strand_info.json') as f1:
        strand_info = json.load(f1)

    # initialize random array for initial_pi
    initial_pi = Utils.create_random_array(DIM_K)

    # read signatures array from BRCA-signatures.npy
    # this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
    signatures_data = [Utils.create_random_array(DIM_M) for i in range(DIM_K)]

    print("Started cross validation for 2'nd type algorithm")

    cross_val_mat = compute_cross_validation_for_total_training_data(strand_info, initial_pi, signatures_data)


# function call
main_algorithm_2_for_1_strand()
