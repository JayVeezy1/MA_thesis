import datetime
import warnings

import shap
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer, \
    accuracy_score, recall_score, average_precision_score, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import  RandomOverSampler, SMOTENC
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
import seaborn as sn

from step_2_preprocessing.preprocessing_functions import get_one_hot_encoding
from step_4_classification.classification_deeplearning import get_DL_auc_score, get_DL_confusion_matrix, \
    split_classification_data_DL, get_sequential_model


def preprocess_for_classification(selected_cohort, features_df, selected_features: list,
                                  selected_dependent_variable: str):
    # Removal of known features_to_remove
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    selected_features = [x for x in selected_features if x not in features_to_remove]

    # todo future research: does it really make sense to keep cluster feature? Or should it be ignored for classification?
    if 'cluster' in selected_cohort.columns:
        selected_features.append('cluster')

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

    # One Hot Encoding: raises recall from 0.56 to 0.58 for XGBOOST
    selected_cohort = get_one_hot_encoding(selected_cohort, categorical_features)

    try:
        selected_cohort.drop(columns='icustay_id', inplace=True)
    except KeyError as e:
        pass

    return selected_cohort  # dependent_variable will be needed and then removed outside


def get_sampled_data(clf, sampling_method, basic_x_train, basic_x_test, basic_y_train, basic_y_test, cohort_title, test_size,
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
            try:
                ros = RandomOverSampler(random_state=1321)
                x_res, y_res = ros.fit_resample(basic_x_train, basic_y_train)
                new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_res, y_res,
                                                                                    test_size=test_size,
                                                                                    random_state=1321)
                x_train_final = new_x_train
                y_train_final = new_y_train
                x_test_final = new_x_test
                y_test_final = new_y_test
                sampling_title = 'random_oversampling'
            except ValueError as e:
                print('WARNING: ValueError occurred. No sampling possible.', e)
                x_train_final = basic_x_train
                y_train_final = basic_y_train
                x_test_final = basic_x_test
                y_test_final = basic_y_test
                sampling_title = 'no_sampling'
        else:
            # checking for categorical features with only columns with 1 or 0 (and 0.5 for stroke_type)
            # because this is already after one-hot encoding. Column names are not in FEATURES_PREPROCESSING table
            categorical_features_positions = []
            for i, column in enumerate(basic_x_train.columns):
                if basic_x_train[column].isin([0, 0.5, 1]).all():
                    categorical_features_positions.append(i)

            smote = SMOTENC(categorical_features=categorical_features_positions,
                            sampling_strategy='auto', random_state=1321, k_neighbors=5)
            # smote = SMOTE(random_state=1321)        # this was the old SMOTE version for only numerical features
            try:
                smote_x_resampled, smote_y_resampled = smote.fit_resample(basic_x_train, basic_y_train)
                new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(smote_x_resampled, smote_y_resampled,
                                                                                    test_size=test_size,
                                                                                    random_state=1321)
                x_train_final = new_x_train
                y_train_final = new_y_train
                x_test_final = new_x_test
                y_test_final = new_y_test
                sampling_title = 'oversampling_smote'
            except ValueError as e:
                print('WARNING: ValueError occurred. No sampling possible.', e)
                x_train_final = basic_x_train
                y_train_final = basic_y_train
                x_test_final = basic_x_test
                y_test_final = basic_y_test
                sampling_title = 'no_sampling'

    elif sampling_method == 'undersampling':  # Undersampling: NearMiss
        # Undersampling: NearMiss
        x_train_final = basic_x_train
        y_train_final = basic_y_train
        x_test_final = basic_x_test
        y_test_final = basic_y_test
        sampling_title = f'undersampling_nearmiss_FAILED'
        versions = [1, 2, 3]
        best_recall = 0
        # try all 3 nearmiss versions
        for version in versions:
            if verbose:
                print(f'STATUS: Implementing NearMiss {version} oversampling for {cohort_title}.')
            nearmiss = NearMiss(version=version)
            nearmiss_x_resampled, nearmiss_y_resampled = nearmiss.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(nearmiss_x_resampled,
                                                                                nearmiss_y_resampled,
                                                                                test_size=test_size, random_state=1321)
            clf.fit(X=new_x_train, y=new_y_train)

            # Choose best recall of all nearmiss versions
            temp_cm = confusion_matrix(new_y_test, clf.predict(new_x_test))
            temp_recall = temp_cm[1][1] / (
                    temp_cm[1][1] + temp_cm[1][0])  # Recall = TruePositives / (TruePositives + FalseNegatives)
            if temp_recall >= best_recall:
                best_recall = temp_recall
                x_train_final = new_x_train
                y_train_final = new_y_train
                x_test_final = new_x_test
                y_test_final = new_y_test
                sampling_title = f'undersampling_nearmiss_v{version}'
    else:
        print(
            f'ERROR: sampling_method {sampling_method} not valid. Choose from options: "no_sampling" or "oversampling" or "undersampling"')
        return None

    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title


def split_classification_data(selected_cohort, cohort_title: str, features_df,
                              selected_features: list,
                              selected_dependent_variable: str, classification_method: str, sampling_method: str,
                              test_size: float,
                              use_grid_search: False,
                              verbose: True):
    # Classification/Prediction on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification(selected_cohort=selected_cohort,
                                           features_df=features_df,
                                           selected_features=selected_features,
                                           selected_dependent_variable=selected_dependent_variable)
    death_df = avg_df[selected_dependent_variable]  # death_label as its own df y_data
    avg_df_filtered = avg_df.drop([selected_dependent_variable], axis=1)  # death_label not inside x_data

    # Create basic training_test_split
    if classification_method == 'RandomForest':
        clf = RandomForestClassifier(random_state=1321, oob_score=True)
    elif classification_method == 'RandomForest_with_gridsearch':
        clf = RandomForestClassifier(random_state=1321, oob_score=True)
    elif classification_method == 'XGBoost':
        clf = XGBClassifier(random_state=1321)                   # todo future research: add and optimize parameters for classifiers
    else:
        print(
            f'ERROR: classification_method "{classification_method}" not valid. Choose from options: "RandomForest" or "XGBoost".')
        return None

    # Split training/test-data
    x_train_basic, x_test_basic, y_train_basic, y_test_basic = train_test_split(avg_df_filtered, death_df,
                                                                                test_size=test_size, random_state=1321)

    # If selected, get over-/under-sampled data
    try:
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title = get_sampled_data(clf,
                                                                                                    sampling_method,
                                                                                                    x_train_basic,
                                                                                                    x_test_basic,
                                                                                                    y_train_basic,
                                                                                                    y_test_basic,
                                                                                                    cohort_title,
                                                                                                    test_size,
                                                                                                    verbose)
    except TypeError as e:
        return None

    # optimize RF classifier
    if (
            classification_method == 'RandomForest' or classification_method == 'RandomForest_with_gridsearch') and use_grid_search:
        clf = grid_search_optimal_RF(clf, x_train_final, y_train_final, verbose)

    # sometimes labels for XGBOOST have to be 'cleaned'
    # from https://stackoverflow.com/questions/71996617/invalid-classes-inferred-from-unique-values-of-y-expected-0-1-2-3-4-5-got
    if classification_method == 'XGBoost':
        le = LabelEncoder()
        y_train_final = le.fit_transform(y_train_final)

    # this is the training step, prediction will be inside the Classification Report
    clf.fit(X=x_train_final, y=y_train_final)

    return clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic


def save_plot_to_file(plot_name, use_case_name, classification_method, cohort_title, sampling_title, selected_feature):
    current_time = datetime.datetime.now().strftime("%H_%M_%S")
    filename = f'./output/{use_case_name}/classification/shapley/{plot_name}_{selected_feature}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png'
    plt.savefig(filename, dpi=600)
    print(f'STATUS: Plot  {plot_name} was saved to {filename}')


def save_plot_to_cache(plot_name, classification_method, cohort_title, sampling_title, selected_feature):
    # Old idea: save SHAP plots in a temp folder and open as image in streamlit
    # But directly plotting with st.pyplot() is possible with streamlit
    filename = f'./web_app/data_upload/temp/{plot_name}_{selected_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
    plt.savefig(filename, dpi=600)
    print(f'STATUS: Plot {plot_name} was saved to {filename}')


@st.cache_data
def get_shapely_values(use_this_function, selected_feature, classification_method, sampling_method, selected_cohort, cohort_title,
                          use_case_name, features_df, selected_features, selected_dependent_variable, show_plot, test_size,
                          use_grid_search, verbose, save_to_cache, save_to_file):
    # calculate the CM, return: CM as dataframe
    if not use_this_function:
        return None, None, None, None

    if classification_method == 'deeplearning_sequential':
        try:
            x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
                selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
                selected_features=selected_features,
                selected_dependent_variable=selected_dependent_variable, test_size=test_size,
                sampling_method=sampling_method, verbose=verbose)
            model, history = get_sequential_model(x_train_final=x_train_final, y_train_final=y_train_final)

            X = x_test_final  # x_test_final | x_train_final
            explainer = shap.DeepExplainer(model=model, data=X)       # todo future research: somehow parameter 'model' does not work here, how calculate shap for DL?
            shap_values = explainer(X=X, check_additivity=True)
        except AssertionError as e:
            st.write(e)
            return None, None, None, None

    else:
        clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
            selected_cohort=selected_cohort,
            cohort_title=cohort_title,
            features_df=features_df,
            selected_features=selected_features,
            selected_dependent_variable=selected_dependent_variable,
            classification_method=classification_method,
            sampling_method=sampling_method,
            test_size=test_size,
            use_grid_search=use_grid_search,
            verbose=verbose)

        X = x_test_final                                        # x_test_final | x_train_final
        explainer = shap.TreeExplainer(model=clf, X=X)          # this creates a TreeObject with empty data, # if model.predict in here, use X = x_train_final for shap_values, otherwise use x_test_final
        shap_values = explainer(X=X, check_additivity=True)     # Constructor of the TreeObject returns an ExplanationObject, it contains base_values, data, feature_names, values
        shap_values_array = explainer.shap_values(X=X, check_additivity=True)

        # either use Explainer=TreeObject(constructor) and then Explainer.values or alternative TreeObject.shap_values
        # based on TreeObject: shap_values_array = explainer.shap_values(X=X, check_additivity=True)

        # Testing if displayed plot works
        if verbose:
            if len(shap_values.shape) == 3:
                try:
                    shap_values.values = shap_values_array[
                        1]  # maybe alternative: for multi-class models use shap_values[0][0] as input
                    shap_values.base_values = shap_values.base_values[0][1]
                except IndexError as e:
                    print('Warning: IndexError occurred for shap_values: ', e)
                    shap_values.values = shap_values_array
                    shap_values.base_values = [shap_values.base_values[1]]

            shap.summary_plot(shap_values, show=False)
            plt.title(f'Beeswarm Plot of Shapley Values for {classification_method}, {sampling_title} on {cohort_title}', wrap=True)
            plt.show()
            plt.close()

        # reduced samples not needed
        # selected_instances = 150
        # reduced_shap_values = shap_values[:selected_instances]      # returns an array of the .values attribute
        # if verbose:
        #     print('CHECK: shap_values: ')
        #     print(reduced_shap_values)

        # Old Approach with one function for each plot to save it locally
        # sample_ind = 20
        # waterfall_plot = shap.plots.waterfall(shap_values[sample_ind], show=False)  # does not work
        # plt.title(f'Waterfall Plot of Shapley Values for {use_case_name}, {classification_method} on {cohort_title}',
        #           wrap=True)
        # if save_to_cache:
        #     save_plot_to_cache(plot_name='waterfall', classification_method=classification_method,
        #                        cohort_title=cohort_title,
        #                        sampling_title=sampling_title, selected_feature=selected_feature)
        # if save_to_file:
        #     save_plot_to_file(plot_name='waterfall', use_case_name=use_case_name,
        #                       classification_method=classification_method, cohort_title=cohort_title,
        #                       sampling_title=sampling_title, selected_feature=selected_feature)
        # if show_plot:
        #     plt.show()
        # plt.close()

    return X, explainer, shap_values, sampling_title


@st.cache_data
def get_confusion_matrix(use_this_function: False, selected_cohort, cohort_title: str,
                         features_df,
                         selected_features: list, selected_dependent_variable: str, classification_method: str,
                         sampling_method: str, use_case_name, save_to_file, test_size, use_grid_search, verbose: True):
    # calculate the CM, return: CM as dataframe
    if not use_this_function:
        return None

    if classification_method == 'deeplearning_sequential':
        cm_df = get_DL_confusion_matrix(selected_cohort, cohort_title, features_df,
                                        selected_features, selected_dependent_variable, classification_method,
                                        sampling_method, use_case_name, test_size, save_to_file, verbose)
        return cm_df

    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, test_size, use_grid_search, verbose)

    # Get CM
    cm: ndarray = confusion_matrix(y_test_basic, clf.predict(x_test_basic))
    # Get CM as table
    try:
        cm_df = pd.DataFrame({
            "predicts_death": {"death": cm[1][1], "no_death": cm[0][1]},
            "predicts_no_death": {"death": cm[1][0], "no_death": cm[0][0]}
        })
    except IndexError as e:
        print(
            f'WARNING: Confusion Matrix does not have all columns/rows. Probably no death cases in {cohort_title} (y_test_final only 0).'
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
        # save CM as seaborn png
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
            annot_kws={"size": 15},
            linewidths=0.5,
            ax=ax1,
            cbar=False,
            cmap='viridis',
            vmin=0,
            vmax=(cm_df['sum_actual']['sum_predicted'] + 20)
            # adding a bit to max value -> not such a strong color difference
        )
        # sn.set(font_scale=3.0)

        # set ticklabels rotation (0 rotation, but with this horizontal)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

        # titles and legends
        plt.tick_params(axis='x', which='major', labelsize=11, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        ax1.set_title(f'{classification_method} on {cohort_title}, {sampling_title}', wrap=True)
        plt.tight_layout()

        # save plot
        if classification_method == 'RandomForest':
            classification_method = 'RF'
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        plt.savefig(
            f'./output/{use_case_name}/classification/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png', dpi=600)
        plt.show()
        plt.close()

        # save CM as .csv - was not necessary
        # cm_filename_string: str = f'./output/{use_case_name}/classification/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        # cm_filename = cm_filename_string.encode()
        # with open(cm_filename, 'w', newline='') as output_file:
        #     cm_df.to_csv(output_file)
        #     print(f'STATUS: cm_df was saved to {cm_filename}')

    return cm_df

@st.cache_data
def get_classification_report(use_this_function: False, display_confusion_matrix: False, selected_cohort,
                              cohort_title: str,
                              features_df,
                              selected_features: list, selected_dependent_variable: str, classification_method: str,
                              sampling_method: str, use_case_name, save_to_file, test_size, use_grid_search: False, verbose: True):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    if display_confusion_matrix:
        cm = get_confusion_matrix(use_this_function=True,  # True | False
                                  classification_method=classification_method,
                                  sampling_method=sampling_method,
                                  selected_cohort=selected_cohort,
                                  cohort_title=cohort_title,
                                  use_case_name=use_case_name,
                                  features_df=features_df,
                                  selected_features=selected_features,
                                  selected_dependent_variable=selected_dependent_variable,
                                  use_grid_search=use_grid_search,
                                  test_size=test_size,
                                  verbose=verbose,
                                  save_to_file=save_to_file)
    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, test_size, use_grid_search, verbose)

    y_pred = clf.predict(x_test_basic)
    report_dict = classification_report(y_test_basic, y_pred, output_dict=True)
    report = pd.DataFrame(report_dict).transpose()
    if verbose:
        print(f'\n CHECK: Classification Report for {classification_method} on {cohort_title}, {sampling_title}:')
        print(report)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification/REPORT_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            report.to_csv(output_file, index=False)
            print(f'STATUS: report was saved to {report_filename}')

    return report


def get_auc_score(use_this_function: False, selected_cohort, cohort_title: str, features_df,
                  selected_features: list, selected_dependent_variable: str, classification_method: str,
                  sampling_method: str, use_case_name, show_plot: False, save_to_file: False, test_size: float,
                  use_grid_search: False,
                  verbose: True):
    # calculate & plot the AUROC, return: auc_score
    # also calculate & plot AUPRC and return: auc_prc_score
    if not use_this_function:
        return None, None, None, None

    if classification_method == 'deeplearning_sequential':
        auc_score, auroc_plot, auc_prc_score, auc_prc_plot = get_DL_auc_score(selected_cohort=selected_cohort, cohort_title=cohort_title,
                                                    features_df=features_df, selected_features=selected_features,
                                                    selected_dependent_variable=selected_dependent_variable,
                                                    classification_method=classification_method,
                                                    sampling_method=sampling_method,
                                                    show_plot=show_plot, use_case_name=use_case_name,
                                                    test_size=test_size,
                                                    save_to_file=save_to_file,
                                                    verbose=verbose)

        return auc_score, auroc_plot, auc_prc_score, auc_prc_plot

    # split_classification_data
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable, classification_method=classification_method,
        sampling_method=sampling_method, test_size=test_size, use_grid_search=use_grid_search, verbose=verbose)

    # Calculate predictions for x_test
    y_pred = clf.predict_proba(x_test_basic)  # Prediction probabilities (= estimated values of prediction)
    try:
        y_pred = y_pred[:, 1]  # Only the probabilities for positive outcome are kept
    except IndexError as e:
        print('Warning: IndexError occurred. y_pred is set as [].', e)
        y_pred = []

    # Get auc_score and auc_prc_score
    if y_test_final.sum() == 0:
        print('WARNING: No death cases in y_test_final. Calculation of auc_score not possible.')
        auc_score = 0
        auc_prc_score = 0
    else:
        # ROC = receiver operating characteristic, AUROC = area under the ROC curve
        try:
            auc_score = round(roc_auc_score(y_test_basic, y_pred), 3)
        except ValueError as e:
            print('Warning: ValueError. auc_score was set to 0. ', e)
            auc_score = 0
        # average precision score = https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        # displays relation between precision to recall
        try:
            auc_prc_score = round(average_precision_score(y_test_basic, y_pred), 3)
        except ValueError as e:
            print('Warning: ValueError. auc_prc_score was set to 0. ', e)
            auc_prc_score = 0
    # print(f'CHECK: {classification_method}: AUROC = %.3f' % auc_score)
    # print(f'CHECK: {classification_method}: average_precision_score = %.3f' % auc_prc_score)

    ## Plot AUC-ROC Curve
    # Get false-positive-rate = x-axis and true-positive-rate = y-axis
    if y_test_basic.sum() == 0:
        print('WARNING: No death cases in y_test_final. Calculation of roc_curve not possible.')
        warnings.filterwarnings(action='ignore',
                                message='No positive samples in y_true, true positive value should be meaningless')  # UndefinedMetricWarning:
        try:
            clf_fpr, clf_tpr, _ = roc_curve(y_test_basic, y_pred)
            auroc_plot = None
        except ValueError as e:
            print('Warning: Value Error occurred. clf_fpr and clf_tpr are set to 0. ', e)
            clf_fpr, clf_tpr, _ = (0, 0, 0)
            auroc_plot = None
    else:
        try:
            clf_fpr, clf_tpr, _ = roc_curve(y_test_basic, y_pred)
            auroc_plot = plt.figure()
            plt.plot(clf_fpr, clf_tpr, label=f'{classification_method} (AUROC = {auc_score})')  # marker='.',
        except ValueError as e:
            print('Warning: Value Error occurred. clf_fpr and clf_tpr are set to 0. ', e)
            auroc_plot = plt.figure()
            clf_fpr, clf_tpr, _ = (0, 0, 0)

    # Add a random predictor line to plot
    random_probs = [0 for _ in range(len(y_test_basic))]
    if y_test_basic.sum() == 0:
        pass
    else:
        try:
            random_auc = roc_auc_score(y_test_basic, random_probs)
            random_fpr, random_tpr, _ = roc_curve(y_test_basic, random_probs)
            plt.plot(random_fpr, random_tpr, linestyle='--', label=f'Random prediction (AUROC = {random_auc})')
        except ValueError as e:
            print('Warning: Value Error occurred. random_auc not plotted. ', e)

    # Plot Settings
    plt.title(f'{classification_method} for {cohort_title} AUROC: {auc_score}, {sampling_title}', wrap=True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        auroc_filename = f'./output/{use_case_name}/classification/AUROC_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png'
        plt.savefig(auroc_filename, dpi=600)
        print(f'STATUS: AUROC was saved to {auroc_filename}')
    if show_plot:
        plt.show()
    plt.close()

    # Plot AUPRC Curve
    try:
        display = PrecisionRecallDisplay.from_predictions(y_test_basic, y_pred,
                                                          name=f'{classification_method} (AUPRC = {auc_prc_score})')
        _ = display.ax_.set_title(f'{classification_method} (AUPRC = {auc_prc_score})')
        plt.title(f"{classification_method} for {cohort_title} AUPRC: {auc_prc_score}, {sampling_title}", wrap=True)
        auc_prc_plot = display.figure_
        if save_to_file:
            current_time = datetime.datetime.now().strftime("%H_%M_%S")
            auprc_filename = f'./output/{use_case_name}/classification/AUPRC_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png'
            plt.savefig(auprc_filename, dpi=600)
            print(f'STATUS: AUPRC was saved to {auprc_filename}')
        if show_plot:
            plt.show()
        plt.close()
        # do not close plt, this throws RunTimeError because TKinter not thread safe https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop#14695007
    except ValueError as e:
        print('Warning: Plotting of PrecisionRecall not possible. ValueError: ', e)
        auc_prc_plot = None

    return auc_score, auroc_plot, auc_prc_score, auc_prc_plot


def get_accuracy(cm_df):
    # (TP + TN) / (TP + FP + FN + TN)
    sum_all_cases = (
            cm_df['predicts_death']['death'] + cm_df['predicts_no_death']['no_death'] + cm_df['predicts_no_death'][
        'death'] + cm_df['predicts_death']['no_death'])
    if sum_all_cases == 0:
        accuracy = 0
    else:
        accuracy = round((cm_df['predicts_death']['death'] + cm_df['predicts_no_death']['no_death']) / sum_all_cases, 3)
    return accuracy


def get_recall(cm_df):
    # TP / (TP + FN)
    if (cm_df['predicts_death']['death'] + cm_df['predicts_no_death']['death']) == 0:
        recall = 0
    else:
        recall = round(
            cm_df['predicts_death']['death'] / (cm_df['predicts_death']['death'] + cm_df['predicts_no_death']['death']),
            3)
    return recall


def get_precision(cm_df):
    # TP / (TP + FP)
    if (cm_df['predicts_death']['death'] + cm_df['predicts_death']['no_death']) == 0:
        precision = 0
    else:
        precision = round(
            cm_df['predicts_death']['death'] / (cm_df['predicts_death']['death'] + cm_df['predicts_death']['no_death']),
            3)
    return precision


def compare_classification_models_on_cohort(use_this_function, use_case_name, features_df, selected_features,
                                            sampling_method, all_cohorts_with_titles, all_classification_methods,
                                            all_dependent_variables, test_size, save_to_file):
    # calculate & plot the AUROC, return: auc_score
    if not use_this_function:
        return None

    classification_models_overview = pd.DataFrame()  # columns: cohort | classification_method | dependent_variable | auc_score -> to add: accuracy | recall

    for title_with_cohort in all_cohorts_with_titles.items():
        for classification_method in all_classification_methods:
            if classification_method == 'RandomForest_with_gridsearch':
                use_grid_search = True
            else:
                use_grid_search = False
            for dependent_variable in all_dependent_variables:
                print(
                    f'STATUS: Calculating auc_score for model settings: {title_with_cohort[0]}, {classification_method}, {dependent_variable}')
                # get auc_score
                auc_score, auroc_plot, auc_prc_score, auc_prc_plot = get_auc_score(use_this_function=True,  # True | False
                                                         classification_method=classification_method,
                                                         sampling_method='oversampling',  # sampling_method,
                                                         # SELECTED_SAMPLING_METHOD  -> currently always oversampling, can also be parameterized
                                                         selected_cohort=title_with_cohort[1],
                                                         cohort_title=f'{title_with_cohort[0]}',
                                                         use_case_name=use_case_name,
                                                         features_df=features_df,
                                                         selected_features=selected_features,
                                                         selected_dependent_variable=dependent_variable,
                                                         show_plot=False,
                                                         verbose=False,
                                                         test_size=test_size,
                                                         use_grid_search=use_grid_search,
                                                         save_to_file=False)
                cm_df = get_confusion_matrix(use_this_function=True,  # True | False
                                             classification_method=classification_method,
                                             sampling_method='oversampling',  # sampling_method,
                                             selected_cohort=title_with_cohort[1],
                                             cohort_title=f'{title_with_cohort[0]}',
                                             use_case_name=use_case_name,
                                             features_df=features_df,
                                             selected_features=selected_features,
                                             selected_dependent_variable=dependent_variable,
                                             test_size=test_size,
                                             use_grid_search=use_grid_search,
                                             verbose=False,
                                             save_to_file=False)
                current_settings = pd.DataFrame([{'cohort': f'{title_with_cohort[0]}',
                                                  'classification_method': classification_method,
                                                  'dependent_variable': dependent_variable,
                                                  'auc_score': auc_score,
                                                  'auc_prc_score': auc_prc_score,
                                                  'accuracy': get_accuracy(cm_df),
                                                  'recall': get_recall(cm_df),
                                                  'precision': get_precision(cm_df)
                                                  }])
                # print('CHECK: Current result: \n', current_settings.to_string(index=False))

                classification_models_overview = pd.concat([classification_models_overview, current_settings],
                                                           ignore_index=True)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/classification/models_overview_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            classification_models_overview.to_csv(output_file, index=False)
            print(f'\n STATUS: classification_models_overview was saved to {filename_string}')
    else:
        print('CHECK: classification_models_overview:')
        print(classification_models_overview)

    return None


def grid_search_optimal_RF(clf, x_train_final, y_train_final, verbose: False):
    # Concept from https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/#:~:text=One%20of%20the%20most%20important,in%20the%20case%20of%20classification.
    # Instantiate the grid search model
    params = {'max_depth': [2, 3, 5, 10],  # removed 20, plot to big
              'min_samples_leaf': [5, 10, 20, 50, 100],  # removed 200, not needed
              'n_estimators': [10, 25, 30, 50, 100, 200]}
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'Recall': make_scorer(recall_score)}
    grid_search = GridSearchCV(estimator=clf,
                               param_grid=params,
                               scoring=scoring,
                               refit='Recall',
                               cv=5,
                               n_jobs=-1, verbose=0)

    # Supress warning does not work somehow
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning)
        grid_search.fit(x_train_final, y_train_final)
        # UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates.

    # Return best settings
    best_score = grid_search.best_score_
    rf_best = grid_search.best_estimator_
    feature_importance_df = pd.DataFrame({
        "feature": x_train_final.columns,
        "importance": rf_best.feature_importances_
    })
    feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

    if verbose:
        print('CHECK GridSearchCV result: best_score (oob_score) for RF after GridSearchCV:',
              best_score)  # out-of-bag score
        print(f'CHECK GridSearchCV result: optimal RF settings:', rf_best)
        print('CHECK GridSearchCV result: Feature Importance: \n', feature_importance_df)

    return rf_best


def plot_random_forest(use_this_function, classification_method, sampling_method, selected_cohort, cohort_title,
                       use_case_name, features_df, selected_features, selected_dependent_variable, show_plot, verbose, test_size,
                       use_grid_search, save_to_file):
    # Calculate GridSearchCV to find optimal RF setting, then plot the RF for feature importance
    if not use_this_function:
        return None

    selected_features.remove('stroke_type')

    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features, selected_dependent_variable,
        classification_method,
        sampling_method, test_size, use_grid_search, verbose)

    # Plot optimal RF
    plt.figure(figsize=(10, 8), dpi=900)  # figsize=(40, 20) makes plot too large
    random_decision_tree = clf.estimators_[5]
    plot_tree(decision_tree=random_decision_tree, feature_names=x_train_final.columns,
              class_names=['death', 'no_death'],
              filled=True)
    # inside each node in the decision_tree plot:
    # feature <= decisive-value,
    # gini = entropy,
    # samples = sum of training-data,
    # value[count(class_0), count(class_1)] with counts of complete-data,
    # decision-of-tree = class_0/class_1

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        plt.savefig(
            f'./output/{use_case_name}/classification/RF_plot_{cohort_title}_{sampling_title}_{current_time}.png')
    if show_plot:
        plt.show()
    plt.close()

    return None


def get_cohort_classified(use_this_function, project_path, classification_method, sampling_method,
                          selected_cohort, cohort_title, use_case_name, features_df, selected_features,
                          selected_dependent_variable, test_size, use_grid_search, verbose, save_to_file):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, test_size, use_grid_search, verbose)

    # Create concatenated classified_cohort
    y_pred: ndarray = clf.predict(x_test_basic)  # do not predict on oversampled data, only train
    y_test_basic_array: ndarray = y_test_basic.to_numpy()

    classified_cohort = x_test_basic
    classified_cohort['class'] = y_test_basic_array  # naming convention from ASDF
    classified_cohort['out'] = y_pred

    # Option to merge ground truths to classified_cohort, not needed
    # dependent_variable_df = pd.DataFrame({'ground_truth_values': y_test_basic_array, 'y_pred': y_pred})
    # classified_cohort: dataframe = x_test_basic.merge(right=dependent_variable_df)  # , axis=1)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        cohort_filename_string: str = f'{project_path}exports/{use_case_name}/classified_cohort_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        cohort_filename = cohort_filename_string.encode()
        with open(cohort_filename, 'w', newline='') as output_file:
            classified_cohort.to_csv(output_file, index=False)

            print(f'STATUS: Classified Cohort was saved to {cohort_filename}')

    return classified_cohort
