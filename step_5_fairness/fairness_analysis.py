import datetime

import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import fairlearn
from pandas.core.interchange import dataframe

from step_4_classification.classification import split_classification_data


# TODO: use aif360 + (or fairlearn? Or even the microsoft package?)
def plot_performance_metrics(clf, x_test_final, y_test_final, selected_protected_attributes):
    ### First Idea: get predictions for each group from clf
    # subgroup_privileged, subgroup_unprivileged = get_subgroups(dataset)
    # 1) Split subgroups of x_test_final into x_privileged and x_unprivileged (also for y_test_final)
    # 2) make clf.predict(x_privileged) and clf.predict(x_unprivileged)
    # 3) compare results to split y_test_final or simply get accuracy for these?
    # 4) Get AUROC curve AND AUPRC (for recall) curve -> does that even depend on the input instances or only on the clf ?
    # for total, privileged, unprivileged instances
    # 5) plot all 3 available AUROC curves in one plot

    ### Second Idea: use already existing predictions inside the ClassificationMetric, count TP etc, calculate metrics
    # use dataset -> or directly the metric
    # use https://aif360.readthedocs.io/en/stable/modules/generated/aif360.metrics.utils.compute_num_TF_PN.html
    # 1. find TP_priv, TN_priv, FP_priv, FN_priv
    # 2. get the same for _unpriv
    # probably also get num_instances
    # 3. calculate Accuracy, Recall etc for each


    # TODO: Try this
    ### Third Idea: use fairlearn MetricFrame to directly plot everything you need
    # https://fairlearn.org/v0.8/user_guide/assessment/plotting.html


    ### Additionally: Check if 7.2.1 makes sense to plot ROC curve for groups
    # https://afraenkel.github.io/fairness-book/content/07-score-functions.html


    ### Additionally: Check if plotting accuracy + any_metric depending on threshold (x-axis) can be done?
    # This would be useful for threshold optimization, but can I even change threshold anywhere?
    # https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb follow this

    return None


def get_fairness_report(use_this_function: False, selected_cohort: dataframe,
                        cohort_title: str,
                        features_df: dataframe,
                        selected_features: list, selected_dependent_variable: str, classification_method: str,
                        sampling_method: str, use_case_name, save_to_file, plot_performance_metrics: False,
                        use_grid_search: False, verbose: True):
    # calculate fairness metrics and return fairness-report
    if not use_this_function:
        return None

    # 0) get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, use_grid_search, verbose)

    # 1) select unprivileged_groups and their respective values
    # IMPORTANT: adjust selected_privileged_classes depending on selected_protected_attributes
    selected_protected_attributes = ['gender', 'ethnicity_1']
    selected_privileged_classes = [[1], [1]]  # privileged: gender=1=male, ethnicity_1=white
    # insurance_1 = self_pay, insurance_4 = private | marital_status_1 = not-single | religion_1 = catholic

    # 2) get an aif360 StandardDataset
    # original_labels = selected_cohort[selected_dependent_variable]
    merged_test_data = x_test_final.merge(right=y_test_final, left_index=True, right_index=True)
    dataset = StandardDataset(df=merged_test_data,
                              label_name=selected_dependent_variable,
                              favorable_classes=[0],  # no death = favorable
                              protected_attribute_names=selected_protected_attributes,
                              privileged_classes=selected_privileged_classes)

    # create dataset_pred
    dataset_pred = dataset.copy()
    predicted_labels = clf.predict(x_test_final)
    dataset_pred.labels = predicted_labels

    # derive privileged groups for all protected_attribute_names
    attr = dataset_pred.protected_attribute_names[0]
    idx = dataset_pred.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    # 3) get metric objs for the report
    classification_metric = ClassificationMetric(dataset=dataset,
                                                 classified_dataset=dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    # 4) Calculate Fairness Metrics
    accuracy = round(classification_metric.accuracy(), 3)
    accuracy_def = 'performance metric for overall accuracy'
    accuracy_expected = 1
    recall = round(classification_metric.recall(), 3)
    recall_def = 'performance metric for TP/True Positives detection'
    recall_expected = 1
    precision = round(classification_metric.precision(), 3)
    precision_def = 'performance metric for TP/Predicted Positives detection'
    precision_expected = 1
    num_instances = round(classification_metric.num_instances(), 0)
    num_instances_def = 'Instances used for prediction'
    num_instances_expected = '-'
    statistical_parity_difference = round(classification_metric.statistical_parity_difference(), 3)
    parity_def = 'Difference between Subgroups'
    parity_expected = 0
    disparate_impact = round(classification_metric.disparate_impact(), 3)
    disparate_def = 'Ratio between Subgroups'
    disparate_expected = 1
    true_positive_rate_difference = round(classification_metric.true_positive_rate_difference(), 3)
    tp_rate_def = 'alias for Equal Opportunity'
    tp_rate_expected = 0
    false_negative_rate_difference = round(classification_metric.false_negative_rate_difference(), 3)
    fp_rate_def = 'False Negative Rate, used for Equalized Odds'
    fp_rate_expected = 0
    average_odds_difference = round(classification_metric.average_odds_difference(), 3)
    average_odds_def = 'alias for Equalized Odds'
    average_odds_expected = 0
    generalized_entropy_index = round(classification_metric.generalized_entropy_index(), 3)
    entropy_def = 'alias for Theil Index with alpha=1'
    entropy_expected = 0
    # OPTIONAL metric:
    # differential_fairness_bias_amplification = round(classification_metric.differential_fairness_bias_amplification(), 3)
    # compares empirical_differential_fairness between original and classified dataset
    # edf = occurrence of positive cases between privileged and unprivileged
    # for the question: does my classifier make the occurrence of positive cases higher?

    # 4) return Fairness Report as print if verbose, save as table if save_files
    report = pd.DataFrame({'Accuracy': [accuracy, accuracy_expected, accuracy_def],
                           'Recall': [recall, recall_expected, recall_def],
                           'Precision': [precision, precision_expected, precision_def],
                           'Number of Instances': [num_instances, num_instances_expected, num_instances_def],
                           'Statistical Parity Difference': [statistical_parity_difference, parity_expected, parity_def],
                           'Disparate Impact Ratio': [disparate_impact, disparate_expected, disparate_def],
                           'True Positive Rate Difference': [true_positive_rate_difference, tp_rate_expected, tp_rate_def],
                           'False Negative Rate Difference': [false_negative_rate_difference, fp_rate_expected, fp_rate_def],
                           'Average Odds Difference': [average_odds_difference, average_odds_expected, average_odds_def],
                           # 'differential_fairness_bias_amplification': [differential_fairness_bias_amplification],
                           'Generalized Entropy Index': [generalized_entropy_index, entropy_expected, entropy_def]
                           })

    if verbose:
        print(f'\n CHECK: Fairness Report for {classification_method} on {cohort_title}, {sampling_title}:')
        print(report.transpose().to_string())

    if plot_performance_metrics:
        plot_performance_metrics(dataset)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")         # removed %d%m%Y_ from date
        attributes_string = ''.join(str(e) for e in selected_protected_attributes)
        report_filename_string: str = f'./output/{use_case_name}/classification/FAIRNESS_{attributes_string}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        # code to export a df
        with open(report_filename, 'w', newline='') as output_file:
            report_export = report.transpose()
            report_export.index.names = ['Metrics']
            report_export.rename(columns={0: attributes_string, 1: 'Optimal Value', 2: 'Information'}, inplace=True)
            report_export.to_csv(output_file, index=True)    # keep index here for metrics titles
            print(f'STATUS: fairness_report was saved to {report_filename}')

    return report
