import datetime
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas.core.interchange import dataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer, \
    accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
import seaborn as sn

# Packages for Deep learning:
import statsmodels.api as sm  # might be necessary for deep learning network
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


# Own Functions
from step_2_preprocessing.preprocessing_functions import get_one_hot_encoding
from step_3_data_analysis.clustering import get_kmeans_clusters, plot_sh_score
from step_3_data_analysis.classification import split_classification_data, get_sampled_data, \
    preprocess_for_classification


def get_classification_report_deeplearning(use_this_function, display_confusion_matrix,
                                           sampling_method, selected_cohort, cohort_title, use_case_name, features_df,
                                           selected_features, selected_dependent_variable, verbose,
                                           save_to_file):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    # if display_confusion_matrix:
    # cm = get_confusion_matrix(use_this_function=True,  # True | False
    #                         classification_method=classification_method,
    #                        sampling_method=sampling_method,
    #                       selected_cohort=selected_cohort,
    #                      cohort_title=cohort_title,
    #                     use_case_name=use_case_name,
    #                    features_df=features_df,
    #                   selected_features=selected_features,
    #                  selected_dependent_variable=selected_dependent_variable,
    #                 use_grid_search=use_grid_search,
    #                verbose=verbose,
    #               save_to_file=save_to_file)

    # tf.keras.layers.GRU(
    #     units,
    #     activation='tanh',
    #     recurrent_activation='sigmoid',
    #     use_bias=True,
    #     kernel_initializer='glorot_uniform',
    #     recurrent_initializer='orthogonal',
    #     bias_initializer='zeros',
    #     kernel_regularizer=None,
    #     recurrent_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     recurrent_constraint=None,
    #     bias_constraint=None,
    #     dropout=0.0,
    #     recurrent_dropout=0.0,
    #     return_sequences=False,
    #     return_state=False,
    #     go_backwards=False,
    #     stateful=False,
    #     unroll=False,
    #     time_major=False,
    #     reset_after=True,
    #     **kwargs
    # )



    # Create basic training_test_split
    x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, sampling_method, verbose)

    # TODO: debug and check if x and y are acceptable format for Sequential()
    # define the keras model
    model = Sequential()
    model.add(Dense(14, input_shape=(10, 19), activation='relu'))       # todo: why 10, 19 ?
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='relu'))         # added this layer - useful?
    model.add(Dense(8, activation='relu'))         # added this layer - useful?
    model.add(Dense(8, activation='relu'))         # added this layer - useful?
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    # this is the training step, prediction will be inside the Classification Report
    model.fit(x=x_train_final, y=y_train_final, epochs=150, batch_size=10)      # change epochs back to 150

    # evaluate the keras model
    #_, accuracy = model.evaluate(X=x_train_final, y=y_train_final)
    #print(accuracy)
    # print('Accuracy: %.2f' % (accuracy * 100))        # ERROR: raise TypeError('Invalid keyword arguments: %s' % (kwargs,))

    y_prediction_values = model.predict(x_test_basic)       # TODO: these should not be needed to round in next step but clearly 0 or 1 - why not? whats wrong here?
    report = classification_report(y_test_basic, y_prediction_values.round())
    if verbose:
        print(f'\n CHECK: Classification Report for deeplearning on {cohort_title}, {sampling_title}:')
        print(report)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification_deeplearning/REPORT_deeplearning_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            output_file.write(report)
            output_file.close()
            print(f'STATUS: deeplearning classification report was saved to {report_filename}')

    return report



def split_classification_data_DL(selected_cohort: dataframe, cohort_title: str, features_df: dataframe,
                              selected_features: list, selected_dependent_variable: str, sampling_method: str,
                            verbose: True):
    # Classification/Prediction on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification(selected_cohort=selected_cohort,
                                           features_df=features_df,
                                           selected_features=selected_features,
                                           selected_dependent_variable=selected_dependent_variable)
    death_df = avg_df[selected_dependent_variable]  # death_label as its own df y_data
    avg_df_filtered = avg_df.drop([selected_dependent_variable], axis=1)  # death_label not inside x_data

    # Split training/test-data
    x_train_basic, x_test_basic, y_train_basic, y_test_basic = train_test_split(avg_df_filtered, death_df,
                                                                                test_size=0.2, random_state=1321)

    # If selected, get over-/under-sampled data
    try:
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title = get_sampled_data_DL(sampling_method,
                                                                                                    x_train_basic,
                                                                                                    x_test_basic,
                                                                                                    y_train_basic,
                                                                                                    y_test_basic,
                                                                                                    cohort_title,
                                                                                                    verbose)
    except TypeError as e:
        return None



    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic



def get_sampled_data_DL(sampling_method, basic_x_train, basic_x_test, basic_y_train, basic_y_test, cohort_title,
                     verbose: True):
    if sampling_method == 'no_sampling':  # no over/undersampling
        x_train_final = basic_x_train
        y_train_final = basic_y_train
        x_test_final = basic_x_test
        y_test_final = basic_y_test
        sampling_title = 'no_sampling'
    elif sampling_method == 'oversampling':  # Oversampling: SMOTE
        if verbose:
            print(f'STATUS: Implementing SMOTE oversampling for {cohort_title}.')

        if basic_y_train.sum() < 5:
            if verbose:
                print(
                    f'WARNING: SMOTE can not be implemented for {cohort_title} because not enough death-cases for train-test-split. '
                    f'At least 5 cases needed. '
                    f'RandomOversampling is implemented instead.')
            ros = RandomOverSampler(random_state=1321)
            x_res, y_res = ros.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_res, y_res,
                                                                                test_size=0.2,
                                                                                random_state=1321)
            x_train_final = new_x_train
            y_train_final = new_y_train
            x_test_final = new_x_test
            y_test_final = new_y_test
            sampling_title = 'random_oversampling'
        else:
            smote = SMOTE(random_state=1321)
            smote_x_resampled, smote_y_resampled = smote.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(smote_x_resampled, smote_y_resampled,
                                                                                test_size=0.2,
                                                                                random_state=1321)
            x_train_final = new_x_train
            y_train_final = new_y_train
            x_test_final = new_x_test
            y_test_final = new_y_test
            sampling_title = 'oversampling_smote'
    else:
        # no undersampling for DL -> would need a clf here to optimize/choose the NearMiss version
        x_train_final = None
        y_train_final = None
        x_test_final = None
        y_test_final = None
        sampling_title = 'unknown_method'

    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title
