import datetime
import warnings

import numpy
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, \
    PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Packages for Deep learning:
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from step_2_preprocessing.preprocessing_functions import get_one_hot_encoding


def preprocess_for_classification_DL(selected_cohort, features_df, selected_features: list,
                                     selected_dependent_variable: str):
    # Removal of known features_to_remove
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    selected_features = [x for x in selected_features if x not in features_to_remove]

    # Removal of other dependent_variables
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)
    selected_cohort = selected_cohort[selected_features].fillna(0)

    # Encoding of categorical features (one hot encoding)
    categorical_features = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    categorical_features = [x for x in categorical_features if x in selected_features]
    selected_cohort = get_one_hot_encoding(selected_cohort,
                                           categorical_features)  # raises recall from 0.56 to 0.58 for XGBOOST

    try:
        selected_cohort.drop(columns='icustay_id', inplace=True)
    except KeyError as e:
        pass

    return selected_cohort  # dependent_variable will be needed and then removed outside


def get_DL_auc_score(selected_cohort, cohort_title, features_df,
                     selected_features, selected_dependent_variable, classification_method,
                     sampling_method, use_case_name, show_plot, save_to_file, verbose):
    # split_classification_data
    x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
        selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
        selected_features=selected_features, selected_dependent_variable=selected_dependent_variable,
        sampling_method=sampling_method, verbose=verbose)

    model, history = get_sequential_model(x_train_final=x_train_final, y_train_final=y_train_final)

    # Calculate predictions for x_test
    y_pred_raw = model.predict(x_test_basic)
    y_pred = [numpy.round(x) for x in y_pred_raw]

    # Get auc_score for probabilities compared to real values
    if y_test_basic.sum() == 0:
        print('WARNING: No death cases in y_test_final. Calculation of auc_score not possible.')
        auc_score = 0
        auc_prc_score = 0
    else:
        # ROC = receiver operating characteristic, AUROC = area under the ROC curve
        auc_score = round(roc_auc_score(y_test_basic, y_pred), 3)
        # average precision score = https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        # displays relation between precision to recall
        auc_prc_score = round(average_precision_score(y_test_basic, y_pred), 3)
    # print(f'CHECK: {classification_method}: AUROC = %.3f' % auc_score)

    # Get false-positive-rate = x-axis and true-positive-rate = y-axis
    if y_test_basic.sum() == 0:
        print('WARNING: No death cases in y_test_final. Calculation of roc_curve not possible.')
        warnings.filterwarnings(action='ignore',
                                message='No positive samples in y_true, true positive value should be meaningless')  # UndefinedMetricWarning:
        clf_fpr, clf_tpr, _ = roc_curve(y_test_basic, y_pred)
    else:
        clf_fpr, clf_tpr, _ = roc_curve(y_test_basic, y_pred)
        plt.plot(clf_fpr, clf_tpr, marker='.', label=f'{classification_method} (AUROC = {auc_score})')

    # Add a random predictor line to plot
    random_probs = [0 for _ in range(len(y_test_basic))]
    if y_test_basic.sum() == 0:
        pass
    else:
        random_auc = roc_auc_score(y_test_basic, random_probs)
        random_fpr, random_tpr, _ = roc_curve(y_test_basic, random_probs)
        plt.plot(random_fpr, random_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % random_auc)

    # Plot Settings
    plt.title(f"{classification_method} for {cohort_title} AUROC: {auc_score}, {sampling_title}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    if save_to_file:
        auroc_filename = f'./output/{use_case_name}/classification_deeplearning/AUROC_{classification_method}_{cohort_title}_{sampling_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png'
        plt.savefig(auroc_filename, dpi=600)
        print(f'STATUS: AUROC was saved to {auroc_filename}')
    if show_plot:
        plt.show()

    # Plot AUPRC Curve
    display = PrecisionRecallDisplay.from_predictions(y_test_basic, y_pred,
                                                      name=f'{classification_method} (AUPRC = {auc_prc_score})')
    _ = display.ax_.set_title(f'{classification_method} (AUPRC = {auc_prc_score})')
    plt.title(f"{classification_method} for {cohort_title} AUPRC: {auc_prc_score}, {sampling_title}", wrap=True)
    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        auprc_filename = f'./output/{use_case_name}/classification_deeplearning/AUPRC_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png'
        plt.savefig(auprc_filename, dpi=600)
        print(f'STATUS: AUPRC was saved to {auprc_filename}')
    if show_plot:
        plt.show()

    return auc_score, auc_prc_score


# plot DL CM from existing CM
def plot_DL_confusion_matrix(cm, cohort_title, classification_method, sampling_title,
                             verbose, save_to_file, use_case_name):
    # Get CM as table
    try:
        cm_df = pd.DataFrame({
            "predicts_death": {"death": cm[1][1], "no_death": cm[0][1]},
            "predicts_no_death": {"death": cm[1][0], "no_death": cm[0][0]}
        })
    except IndexError as e:
        print(
            f'WARNING: Confusion Matrix does not have all columns/rows. Probably no death cases in {cohort_title} (y_test_basic only 0).'
            f'Calculation of Classification Report is not possible for this cohort.')
        cm_df = pd.DataFrame({
            "predicts_death": {"death": 0, "no_death": 0},
            # use 0 instead of np.nan -> will be caught in calculation of accuracy, recall, etc.
            "predicts_no_death": {"death": 0, "no_death": 0}
        })

    if verbose:
        print(f'\n Confusion Matrix for {classification_method} on {cohort_title}, {sampling_title}:')
        print(cm_df)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        # save CM as seaborn png
        cmap = 'viridis'
        fig1, ax1 = plt.subplots()

        # add totals to cm_df
        sum_col = []
        for c in cm_df.columns:
            sum_col.append(cm_df[c].sum())
        sum_lin = []
        for item_line in cm_df.iterrows():
            sum_lin.append(item_line[1].sum())
        cm_df['sum_actual'] = sum_lin
        sum_col.append(np.sum(sum_lin))
        cm_df.loc['sum_predicted'] = sum_col

        # create seaborn heatmap
        ax1 = sn.heatmap(
            data=cm_df,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 10},
            linewidths=0.5,
            ax=ax1,
            cbar=False,
            cmap=cmap,
            vmin=0,
            vmax=(cm_df['sum_actual']['sum_predicted'] + 20)
            # adding a bit to max value -> not such a strong color difference
        )

        # set ticklabels rotation (0 rotation, but with this horizontal)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

        # titles and legends
        plt.tick_params(axis='x', which='major', labelsize=11, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        ax1.set_title(f"{classification_method} on {cohort_title}, {sampling_title}", wrap=True)
        plt.tight_layout()

        # save plot
        if classification_method == 'RandomForest':
            classification_method = 'RF'
        plt.savefig(
            f'./output/{use_case_name}/classification_deeplearning/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png',
            dpi=600)
        plt.show()
        plt.close()

        # save CM as .csv -> never needed
        # cm_filename_string: str = f'./output/{use_case_name}/classification/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        # cm_filename = cm_filename_string.encode()
        # with open(cm_filename, 'w', newline='') as output_file:
        #     cm_df.to_csv(output_file)
        #     print(f'STATUS: cm_df was saved to {cm_filename}')

    return cm_df


# get DL from raw data
def get_DL_confusion_matrix(selected_cohort, cohort_title, features_df, selected_features, selected_dependent_variable,
                            classification_method, sampling_method, use_case_name, save_to_file, verbose):
    # get_classification_basics
    x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
        selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable,
        sampling_method=sampling_method, verbose=verbose)

    model, history = get_sequential_model(x_train_final=x_train_final, y_train_final=y_train_final)

    y_pred = model.predict(x=x_test_basic,
                           batch_size=128).round()  # round() needed to get from sigmoid probability to class value 0 or 1
    cm: ndarray = confusion_matrix(y_test_basic, y_pred)

    cm_df = plot_DL_confusion_matrix(cm, cohort_title, classification_method, sampling_title,
                                     verbose, save_to_file, use_case_name)

    return cm_df


# Optional idea: use GPU for Deep Learning model for more efficient and faster computation
# 1) directly with tensorflow the GPU: https://www.tensorflow.org/guide/gpu
# 2) installing theano for GPU: https://theano-pymc.readthedocs.io/en/latest/install_windows.html
# 3) pybinding for miniconda + pip: https://docs.pybinding.site/en/stable/install/quick.html#troubleshooting
# Alternative OpenBLas: https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio must also be setup in Visual Studio
def get_sequential_model(x_train_final, y_train_final):
    # define the keras model
    # Optional idea: use tf.keras.layers.GRU nodes for Deep Learning model instead of Sequential() model https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
    model = Sequential()  # basic keras model for deep learning network
    model.add(Dense(16, input_shape=(None, len(x_train_final.columns)), activation='relu'))
    model.add(Dense(12, activation='relu'))  # amount and setup of layers can be changed
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tensorflow.keras.metrics.Recall()])
    # alternative optimizer: 'sgd'
    # alternative use recall as metric: metrics=[tf.keras.metrics.Recall(thresholds=0)]

    # fit the keras model on the dataset + predict values y_pred
    history = model.fit(x=x_train_final, y=y_train_final, epochs=175, batch_size=32)

    return model, history


def get_classification_report_deeplearning(use_this_function, sampling_method, selected_cohort, cohort_title,
                                           use_case_name, features_df,
                                           selected_features, selected_dependent_variable, verbose,
                                           save_to_file):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    if verbose:
        print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))  # output = 0

    # Create basic training_test_split
    x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
        selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable,
        sampling_method=sampling_method, verbose=verbose)

    model, history = get_sequential_model(x_train_final=x_train_final, y_train_final=y_train_final)
    y_pred = model.predict(x=x_test_basic,
                           batch_size=128).round()  # round() needed to get from sigmoid probability to class value 0 or 1

    # Get complete classification_report
    report = classification_report(y_true=y_test_basic, y_pred=y_pred)

    # Get recall value
    recall_object = tensorflow.keras.metrics.Recall()
    recall_object.update_state(y_true=y_test_basic, y_pred=y_pred)
    recall_value = recall_object.result().numpy()

    if verbose:
        print(f'\n CHECK: Classification Report for deeplearning on {cohort_title}, {sampling_title}:')
        print(report)
        print(f'\n CHECK: recall_object:')
        print(recall_value)

    if save_to_file:
        # Save DL classification_report
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification_deeplearning/REPORT_deeplearning_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            output_file.write(report)
            output_file.close()
            print(f'STATUS: deeplearning classification report was saved to {report_filename}')

        # Save model configurations plot
        loss_and_metrics = model.evaluate(x=x_test_basic, y=y_test_basic, batch_size=128)
        fig, ax = plt.subplots()
        loss_color = '#B00000'
        ax.plot(history.history['loss'], color=loss_color)  # , marker=".")
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel('loss', color=loss_color, fontsize=12)
        ax2 = ax.twinx()  # twin object for second y-axis
        accuracy_color = '#0000B0'
        ax2.plot(history.history['accuracy'], color=accuracy_color)  # , marker=".")
        ax2.set_ylabel('accuracy', color=accuracy_color, fontsize=12)
        plt.title(
            f'Loss({round(loss_and_metrics[0], 2)}) and Accuracy({round(loss_and_metrics[1], 2)}) for DL Model on {cohort_title}',
            wrap=True)
        plt.savefig(
            f'./output/{use_case_name}/classification_deeplearning/MODEL_deeplearning_{cohort_title}_{sampling_title}_{current_time}.png',
            bbox_inches='tight',
            dpi=600)
        plt.show()
        plt.close()

        # Plot CM
        cm: ndarray = confusion_matrix(y_test_basic, y_pred)
        plot_DL_confusion_matrix(cm=cm, cohort_title=cohort_title,
                                 classification_method='deeplearning_sequential', sampling_title=sampling_title,
                                 verbose=verbose, save_to_file=save_to_file, use_case_name=use_case_name)

    return report


def split_classification_data_DL(selected_cohort, cohort_title: str, features_df,
                                 selected_features: list, selected_dependent_variable: str, sampling_method: str,
                                 verbose: True):
    # Classification/Prediction on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification_DL(selected_cohort=selected_cohort,
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
