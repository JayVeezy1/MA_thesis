import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas.core.interchange import dataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
import seaborn as sn


def preprocess_for_classification(selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                                  selected_dependent_variable: str):
    # Preprocessing for Clustering: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)  # keeping selected_dependent_variable for clustering?
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    temp_selected_features = selected_features.copy()
    temp_selected_features.remove(selected_dependent_variable)
    # print(f'CHECK: {len(temp_selected_features)} features used for Classification: ', temp_selected_features)  # dependent_variable will be removed outside

    return selected_cohort[selected_features].fillna(0)


def get_sampled_data(clf, sampling_method, basic_x_train, basic_x_test, basic_y_train, basic_y_test, cohort_title,
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
                                                                                test_size=0.2, random_state=1321)
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


def split_classification_data(selected_cohort: dataframe, cohort_title: str, features_df: dataframe, selected_features: list,
                              selected_dependent_variable: str, classification_method: str, sampling_method: str,
                              verbose: True):
    # Classification/Prediction on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification(selected_cohort, features_df, selected_features, selected_dependent_variable)
    death_df = avg_df[selected_dependent_variable]  # death_label as its own df y_data
    avg_df_filtered = avg_df.drop([selected_dependent_variable], axis=1)  # death_label not inside x_data

    # Create basic training_test_split
    if classification_method == 'RandomForest':
        clf = RandomForestClassifier(random_state=1321)  # old settings: n_estimators=100, max_depth=5, bootstrap=True,
    elif classification_method == 'XGBoost':
        clf = XGBClassifier()
    else:
        print(
            f'ERROR: classification_method "{classification_method}" not valid. Choose from options: "RandomForest" or "XGBoost".')
        return None

    # Split training/test-data
    x_train_basic, x_test_basic, y_train_basic, y_test_basic = train_test_split(avg_df_filtered, death_df,
                                                                                test_size=0.2, random_state=1321)
    # If selected, get over-/under-sampled data
    try:
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title = get_sampled_data(clf,
                                                                                                    sampling_method,
                                                                                                    x_train_basic,
                                                                                                    x_test_basic,
                                                                                                    y_train_basic,
                                                                                                    y_test_basic,
                                                                                                    cohort_title,
                                                                                                    verbose)
    except TypeError as e:
        return None

    # this is the training step, prediction will be inside the Classification Report
    clf.fit(X=x_train_final, y=y_train_final)

    return clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic


def get_confusion_matrix(use_this_function: False, selected_cohort: dataframe, cohort_title: str, features_df: dataframe,
                         selected_features: list, selected_dependent_variable: str, classification_method: str,
                         sampling_method: str, use_case_name, save_to_file, verbose: True):
    # calculate the CM, return: CM as dataframe
    if not use_this_function:
        return None

    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, verbose)

    # Get CM
    cm: ndarray = confusion_matrix(y_test_basic,
                                   clf.predict(x_test_basic))  # important: use test_basic here, not the sampled version
    # Get CM as table
    cm_df = pd.DataFrame({
        "predicts_true": {"is_true": cm[1][1], "is_false": cm[0][1]},
        "predicts_false": {"is_true": cm[1][0], "is_false": cm[0][0]}
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
            vmax=(cm_df['sum_actual']['sum_predicted'] + 20)        # adding a bit to max value -> not such a strong color difference
        )

        # set ticklabels rotation (0 rotation, but with this horizontal)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

        # titles and legends
        plt.tick_params(axis='x', which='major', labelsize=11, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        ax1.set_title(f"{classification_method} on {cohort_title}, {sampling_title}")
        plt.tight_layout()

        # save plot
        if classification_method == 'RandomForest':
            classification_method = 'RF'
        plt.savefig(f'./output/{use_case_name}/classification/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}_new.png')
        plt.show()
        plt.close()

        # save CM as .csv
        cm_filename_string: str = f'./output/{use_case_name}/classification/CM_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        cm_filename = cm_filename_string.encode()
        with open(cm_filename, 'w', newline='') as output_file:
            cm_df.to_csv(output_file)
            print(f'STATUS: cm_df was saved to {cm_filename}')

    return cm_df


def get_classification_report(use_this_function: False, selected_cohort: dataframe, cohort_title: str,
                              features_df: dataframe,
                              selected_features: list, selected_dependent_variable: str, classification_method: str,
                              sampling_method: str, use_case_name, save_to_file, verbose: True):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    # get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, verbose)

    report = classification_report(y_test_basic, clf.predict(x_test_basic))
    if verbose:
        print(f'\n CHECK: Classification Report for {classification_method} on {cohort_title}, {sampling_title}:')
        print(report)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification/REPORT_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            output_file.write(report)
            output_file.close()
            print(f'STATUS: report was saved to {report_filename}')

    return report


def get_auc_score(use_this_function: False, selected_cohort: dataframe, cohort_title: str, features_df: dataframe,
                  selected_features: list, selected_dependent_variable: str, classification_method: str,
                  sampling_method: str, use_case_name, show_plot: False, save_to_file: False, verbose: True):
    # calculate & plot the AUROC, return: auc_score
    if not use_this_function:
        return None

    # split_classification_data
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, verbose)

    # Calculate predictions for x_test
    clf_probs = clf.predict_proba(x_test_basic)  # Prediction probabilities (= estimated values of prediction)
    clf_probs = clf_probs[:, 1]  # Only the probabilities for positive outcome are kept

    # Get auc_score for probabilities compared to real values
    auc_score = round(roc_auc_score(y_test_basic,
                                    clf_probs),
                      3)  # ROC = receiver operating characteristic, AUROC = area under the ROC curve
    # print(f'CHECK: {classification_method}: AUROC = %.3f' % auc_score)

    # Get false-positive-rate = x-axis and true-positive-rate = y-axis
    clf_fpr, clf_tpr, _ = roc_curve(y_test_basic, clf_probs)
    plt.plot(clf_fpr, clf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % auc_score)

    # Add a random predictor line to plot
    random_probs = [0 for _ in range(len(y_test_basic))]
    random_auc = roc_auc_score(y_test_basic, random_probs)
    random_fpr, random_tpr, _ = roc_curve(y_test_basic, random_probs)
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % random_auc)

    # Plot Settings
    plt.title(f"{classification_method} for {cohort_title} AUROC: {auc_score}, {sampling_title}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    if save_to_file:
        auroc_filename = f'./output/{use_case_name}/classification/AUROC_{classification_method}_{cohort_title}_{sampling_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png'
        plt.savefig(auroc_filename)
        print(f'STATUS: AUROC was saved to {auroc_filename}')

    if show_plot:
        plt.show()

    return auc_score


def get_accuracy(cm_df):
    # (TP + TN) / (TP + FP + FN + TN)
    return round((cm_df['predicts_true']['is_true'] + cm_df['predicts_false']['is_false']) / (cm_df.to_numpy().sum()), 3)


def get_recall(cm_df):
    # TP / (TP + FN)
    return round(cm_df['predicts_true']['is_true'] / (cm_df['predicts_true']['is_true'] + cm_df['predicts_false']['is_true']), 3)


def get_precision(cm_df):
    # TP / (TP + FP)
    return round(cm_df['predicts_true']['is_true'] / (cm_df['predicts_true']['is_true'] + cm_df['predicts_true']['is_false']), 3)


def compare_classification_models_on_cohort(use_this_function, use_case_name, features_df, selected_features,
                                            all_cohorts_with_titles,
                                            all_classification_methods,
                                            all_dependent_variables, save_to_file):
    # calculate & plot the AUROC, return: auc_score
    if not use_this_function:
        return None

    classification_models_overview: dataframe = pd.DataFrame()  # columns: cohort | classification_method | dependent_variable | auc_score -> to add: accuracy | recall

    for title_with_cohort in all_cohorts_with_titles.items():
        for classification_method in all_classification_methods:
            for dependent_variable in all_dependent_variables:
                print(
                    f'STATUS: Calculating auc_score for model settings: {title_with_cohort[0]}, {classification_method}, {dependent_variable}')
                # get auc_score
                auc_score = get_auc_score(use_this_function=True,  # True | False
                                          classification_method=classification_method,
                                          sampling_method='oversampling',
                                          # SELECTED_SAMPLING_METHOD  -> currently always oversampling, can also be parameterized
                                          selected_cohort=title_with_cohort[1],
                                          cohort_title=f'{title_with_cohort[0]}',
                                          use_case_name=use_case_name,
                                          features_df=features_df,
                                          selected_features=selected_features,
                                          selected_dependent_variable=dependent_variable,
                                          show_plot=False,
                                          verbose=False,
                                          save_to_file=False
                                          )
                cm_df: dataframe = get_confusion_matrix(use_this_function=True,  # True | False
                                                        classification_method=classification_method,
                                                        sampling_method='oversampling',  # SELECTED_SAMPLING_METHOD
                                                        selected_cohort=title_with_cohort[1],
                                                        cohort_title=f'{title_with_cohort[0]}',
                                                        use_case_name=use_case_name,
                                                        features_df=features_df,
                                                        selected_features=selected_features,
                                                        selected_dependent_variable=dependent_variable,
                                                        verbose=False,
                                                        save_to_file=False
                                                        )
                current_settings = pd.DataFrame([{'cohort': f'{title_with_cohort[0]}',
                                                  'classification_method': classification_method,
                                                  'dependent_variable': dependent_variable,
                                                  'auc_score': auc_score,
                                                  'accuracy': get_accuracy(cm_df),
                                                  'recall': get_recall(cm_df),
                                                  'precision': get_precision(cm_df)
                                                  }])
                # print('CHECK: Current result: \n', current_settings.to_string(index=False))

                classification_models_overview = pd.concat([classification_models_overview, current_settings],
                                                           ignore_index=True)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/classification/models_overview_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            classification_models_overview.to_csv(output_file, index=False)
            print(f'\n STATUS: classification_models_overview was saved to {filename_string}')
    else:
        print('CHECK: classification_models_overview:')
        print(classification_models_overview)

    return None


def compare_classification_models_on_clusters(use_this_function, use_case_name, features_df, selected_features,
                                              selected_cohort, all_classification_methods, all_dependent_variables, save_to_file):

    # step 1: get the ideal cluster count for this selected_cohort

    # step 2: get these clusters into a list

    # step 3: for each cluster get prediction quality


    # problem: does it even make sense to compare clusters cross-model? probably not. cohort has to be the same. classification_method & dependent_variable can be changed.


    return None