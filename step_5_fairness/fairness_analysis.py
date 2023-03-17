import datetime

import pandas as pd
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
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
    # todo: check if problem occurrs with factorized categorical features. Because they are transformed into their own columns like dbsource_1
    # IMPORTANT: adjust selected_privileged_classes depending on selected_protected_attributes
    selected_protected_attributes = ['gender']
    selected_privileged_classes = [[1]]  # male is mapped to 1, is privileged
    # selected_privileged_classes = [[gender = 1], [ethnicity = 1], [insurance = 1, 2, 3], [...] ]

    # 2) get an aif360 StandardDataset
    # idea: https://stackoverflow.com/questions/64506977/calculate-group-fairness-metrics-with-aif360/64543058#64543058
    original_labels = selected_cohort[selected_dependent_variable]
    # todo: make certain correct index used for merge, selected cohort has all patients, but with inner-join only the ones in trained cohort are kept?
    merged_train_data = x_train_final.merge(right=original_labels, left_index=True, right_index=True)
    dataset = StandardDataset(df=merged_train_data,
                              label_name=selected_dependent_variable,
                              favorable_classes=[0],  # no death = favorable
                              protected_attribute_names=selected_protected_attributes,
                              privileged_classes=selected_privileged_classes)

    # create dataset_pred
    dataset_pred = dataset.copy()
    predicted_labels = clf.predict(X=x_train_final)
    dataset_pred.labels = predicted_labels          # TODO: Accuracy is 1.0 -> check if this step was done correctly and dependent variable column was overwritten? Inplace??

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

    metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    # 3) calculate important metrics: demographic_parity_metric, equalized_odds_metric
    equal_opportunity_difference = classification_metric.equal_opportunity_difference()
    accuracy = classification_metric.accuracy()

    statistical_parity_difference = metric_pred.statistical_parity_difference()  # todo: why use metric_pred here? possible with classification_metric?

    average_abs_odds_difference = classification_metric.average_abs_odds_difference()
    disparate_impact = classification_metric.disparate_impact()
    average_odds_difference = classification_metric.average_odds_difference()

    # 4) return Fairness Report as print if verbose, save as table if save_files
    report = pd.DataFrame({'accuracy': [accuracy],
                           'statistical_parity_difference': [statistical_parity_difference],
                           'average_abs_odds_difference': [average_abs_odds_difference],
                           'disparate_impact': [disparate_impact],
                           'equal_opportunity_difference': [equal_opportunity_difference],
                           'average_odds_difference': [average_odds_difference]})
    if verbose:
        print(f'\n CHECK: Fairness Report for {classification_method} on {cohort_title}, {sampling_title}:')
        print(report)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification/FAIRNESS_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        # code to export a df
        with open(report_filename, 'w', newline='') as output_file:
            report.to_csv(output_file, index=False)
            print(f'STATUS: fairness_report was saved to {report_filename}')

        # code to export a dict
        # with open(report_filename, 'w', newline='') as output_file:
        #   for key in report.keys():
        #      output_file.write("%s,%s\n" % (key, report[key]))
        # output_file.close()
        # print(f'STATUS: report was saved to {report_filename}')

    return report
