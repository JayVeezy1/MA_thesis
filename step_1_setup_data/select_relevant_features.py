import csv
import os
from operator import countOf

import pandas as pd
from os.path import isfile

from matplotlib import pyplot as plt


def get_selected_features(project_path, use_case_name) -> list:
    """
    This function selects the features that will be used for the final_dataset

    :param project_path:
    :param use_case_name:
    :param patient_list: list of all patients (from .csv files)
    :return: None
    """
    # IMPORT
    raw_feature_count_dict: dict = {}
    patient_count: int = 0
    for file in os.listdir(f'{project_path}/exports/{use_case_name}/'):
        if isfile(f'{project_path}/exports/{use_case_name}/{file}') and not file == '0_patient_cohort.csv':
            patient_count += 1
            header = pd.read_csv(f'{project_path}/exports/{use_case_name}/{file}', nrows=0)  # only get header

            for feature in header:
                if feature in raw_feature_count_dict.keys():
                    raw_feature_count_dict[feature] += 1
                else:
                    raw_feature_count_dict[feature] = 1  # create with value 1


    # SORTING
    sorted_feature_count_list: list = sorted(raw_feature_count_dict.items(),
                                             key=lambda x: x[1],
                                             reverse=True)  # sorted() creates a list of tuples(), reverse = descending
    sorted_feature_count_dict: dict = dict(sorted_feature_count_list)
    # ('CHECK: The following dictionary offers an overview of the occurrence of features in the dataset. '
    #       'It can be used to select relevant features for the following analysis.')
    # print(sorted_feature_count_dict)
    print(f'Count of patients: {patient_count}, '
          f'Count of features with occurrence for all patients: {countOf(sorted_feature_count_dict.values(), patient_count)}, '
          f'Count of all available features: {len(sorted_feature_count_dict)} ')


    # todo FILTERING
    # manually remove features that are general patient info (will be added again later)
    general_features: list = ['charttime', 'gender', 'age']
    # manually remove features that are known and important from prev research (will be added again later)
    important_features: list = ['heart rate']
    # manually remove features that can be safely dismissed (without medical knowledge)
    unimportant_features: list = ['Nasal Swab']

    filtered_features_count_list: list = []
    for feature_value_tuple in sorted_feature_count_list:
        if not feature_value_tuple[0] in general_features \
                and not feature_value_tuple[0] in important_features \
                and not feature_value_tuple[0] in unimportant_features:
            if (feature_value_tuple[1] / patient_count) >= 0.8:                        # todo 80% threshold but PROBLEM: these might actually be very important
                filtered_features_count_list.append(feature_value_tuple)
    filtered_feature_count_dict: dict = dict(filtered_features_count_list)
    print(f'Count of all remaining features depending on occurrence: {len(filtered_features_count_list)}')

    # Plot the filtered features that occur for all patients
    feature_name, feature_count = zip(*filtered_features_count_list[:(countOf(filtered_feature_count_dict.values(), patient_count) + 10)])
    plt.plot(feature_name, feature_count)
    plt.xticks(rotation=45, ha='right')
    plt.show()


    # Add again the general_features and important_features to final feature-list
    selected_features: list = general_features
    selected_features.extend(important_features)
    selected_features.extend(filtered_features_count_list[:20])             # only keep the features that were selected because of occurrence > 80%

    print(f'Count of final selected features: {len(selected_features)}')

    print('STATUS: Finished get_feature_selection.')

    # Todo Goal:
    # feature-overview-table (first step for feature-selection):
    # label | item_id | count | variable_type (categorical(only 1 row) or continuous (average, min, max)) | selected (selected_general_patient_data or selected_because_research or selected_count_over_xx) |  | removed

    # todo: decide how to use selected_features in rest of the code? might be good to save with other supplements as csv

    return selected_features


def export_final_dataset(project_path, use_case_name, selected_features: list):
    # step 1: load each patient
    for file in os.listdir(f'{project_path}/exports/{use_case_name}/'):
        if isfile(f'{project_path}/exports/{use_case_name}/{file}') and not file == '0_patient_cohort.csv':
            patient_df = pd.read_csv(f'{project_path}/exports/{use_case_name}/{file}')

            filtered_patient_df = patient_df(columns=selected_features)     # todo somehow like this

            # step 2: export the patient but filtered with selected_features (and maybe sort these too?) into a new folder

    return None
