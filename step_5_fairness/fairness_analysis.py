import datetime

import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import fairlearn
from pandas.core.interchange import dataframe

from step_4_classification.classification import split_classification_data


# TODO: use aif360 + (or fairlearn? Or even the microsoft package?)
def get_fairness_report(use_this_function: False, selected_cohort: dataframe,
                        cohort_title: str,
                        features_df: dataframe,
                        selected_features: list, selected_dependent_variable: str, classification_method: str,
                        sampling_method: str, use_case_name, save_to_file, use_grid_search: False, verbose: True):
    # calculate fairness metrics and return fairness-report
    if not use_this_function:
        return None

    # 0) get_classification_basics
    clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, classification_method, sampling_method, use_grid_search, verbose)

    # 1) select unprivileged_groups and their respective values
    # Advantage: check fairness of exactly one of the categories (1 is positive, 0 is not)
    # Disadvantage: not check influence of the complete feature or multiple values of this feature
    # Solution: use ethnicity_1 AND ethnicity_3 (for example) together to check multiple ethnicity_values

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

    # Alternative for metrics (only for dataset_pred)
    # metric_pred = BinaryLabelDatasetMetric(dataset_pred,
    #                                       unprivileged_groups=unprivileged_groups,
    #                                       privileged_groups=privileged_groups)

    # 4) Calculate Fairness Metrics
    accuracy = round(classification_metric.accuracy(), 3)
    recall = round(classification_metric.recall(), 3)
    precision = round(classification_metric.precision(), 3)
    num_instances = round(classification_metric.num_instances(), 0)

    statistical_parity_difference = round(classification_metric.statistical_parity_difference(), 3)
    disparate_impact = round(classification_metric.disparate_impact(), 3)
    true_positive_rate_difference = round(classification_metric.true_positive_rate_difference(), 3)       # = equal_opportunity_difference
    false_negative_rate_difference = round(classification_metric.false_negative_rate_difference(), 3)
    average_odds_difference = round(classification_metric.average_odds_difference(), 3)                 # = equalized_odds_difference

    # differential_fairness_bias_amplification = round(classification_metric.differential_fairness_bias_amplification(), 3)
    # compares empirical_differential_fairness between original and classified dataset
    # edf = occurrence of positive cases between privileged and unprivileged
    # for the question: does my classifier make the occurrence of positive cases higher?
    generalized_entropy_index = round(classification_metric.generalized_entropy_index(), 3)     # alias: theil_index

    # 4) return Fairness Report as print if verbose, save as table if save_files
    report = pd.DataFrame({'accuracy': [accuracy],
                           'recall': [recall],
                           'precision': [precision],
                           'num_instances': [num_instances],
                           'statistical_parity_difference': [statistical_parity_difference],
                           'disparate_impact': [disparate_impact],
                           'true_positive_rate_difference': [true_positive_rate_difference],
                           'false_negative_rate_difference': [false_negative_rate_difference],
                           'average_odds_difference': [average_odds_difference],
                           # 'differential_fairness_bias_amplification': [differential_fairness_bias_amplification],
                           'generalized_entropy_index': [generalized_entropy_index]
                           })

    # TODO: find way to plot metrics
    # especially also accurracy etc. for total, male, female (or alternatives)
    # https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb follow this
    # Check: how to combine ethnicity + gender (or others)

    if verbose:
        print(f'\n CHECK: Fairness Report for {classification_method} on {cohort_title}, {sampling_title}:')
        print(report.transpose().to_string())

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")         # removed %d%m%Y_ from date
        attributes_string = ''.join(str(e) for e in selected_protected_attributes)
        report_filename_string: str = f'./output/{use_case_name}/classification/FAIRNESS_{attributes_string}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        # code to export a df
        with open(report_filename, 'w', newline='') as output_file:
            report_export = report.transpose()
            report_export.rename(columns={0: attributes_string}, inplace=True)
            report_export.to_csv(output_file, index=True)    # keep index here for metrics titles
            print(f'STATUS: fairness_report was saved to {report_filename}')

        # code to export a dict
        # with open(report_filename, 'w', newline='') as output_file:
        #   for key in report.keys():
        #      output_file.write("%s,%s\n" % (key, report[key]))
        # output_file.close()
        # print(f'STATUS: report was saved to {report_filename}')

    return report
