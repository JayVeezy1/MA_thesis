import datetime

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from fairlearn.metrics import MetricFrame, selection_rate, count
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix

from step_4_classification.classification import split_classification_data
from step_4_classification.classification_deeplearning import get_sequential_model, split_classification_data_DL


def generalized_entropy_index(b, alpha=2):
    r"""Generalized entropy index measures inequality over a population.
    from: https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py#L963-L987

    Args:
        b (array-like): Parameter over which to calculate the entropy index.
        alpha (scalar): Parameter that regulates the weight given to distances
            between values at different parts of the distribution. A value of 0
            is equivalent to the mean log deviation, 1 is the Theil index, and 2
            is half the squared coefficient of variation.
    """
    if alpha == 0:
        return -(np.log(b / b.mean()) / b.mean()).mean()
    elif alpha == 1:
        # moving the b inside the log allows for 0 values
        return (np.log((b / b.mean())**b) / b.mean()).mean()
    else:
        return ((b / b.mean())**alpha - 1).mean() / (alpha * (alpha - 1))

def generalized_entropy_error(y_true, y_pred, alpha=2, pos_label=1):
    #                           sample_weight=None):
    r"""Compute the generalized entropy.
    from: https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py#L963-L987

    Generalized entropy index is proposed as a unified individual and
    group fairness measure in [#speicher18]_.

    Uses :math:`b_i = \hat{y}_i - y_i + 1`. See
    :func:`generalized_entropy_index` for details.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        alpha (scalar, optional): Parameter that regulates the weight given to
            distances between values at different parts of the distribution. A
            value of 0 is equivalent to the mean log deviation, 1 is the Theil
            index, and 2 is half the squared coefficient of variation.
        pos_label (scalar, optional): The label of the positive class.

    See also:
        :func:`generalized_entropy_index`

    References:
        .. [#speicher18] `T. Speicher, H. Heidari, N. Grgic-Hlaca,
           K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar, "A Unified
           Approach to Quantifying Algorithmic Unfairness: Measuring Individual
           and Group Unfairness via Inequality Indices," ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           2018. <https://dl.acm.org/citation.cfm?id=3220046>`_
    """
    b = 1 + (y_pred == pos_label) - (y_true == pos_label)
    return generalized_entropy_index(b, alpha=alpha)

def get_rates_from_cm(cm):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
    tn = cm.loc[0, 'predicted_0']
    fn = cm.loc[1, 'predicted_0']
    fp = cm.loc[0, 'predicted_1']
    tp = cm.loc[1, 'predicted_1']

    true_positive_rate = tp / (tp + fn)         # = sensitivity
    false_positive_rate = fp / (fp + tn)        # = 1 - specificity
    # true_negative_rate = tn / (tn + fp)
    # false_negative_rate = fn / (fn + tp)

    return true_positive_rate, false_positive_rate

def create_performance_metrics_plot(y_pred, y_true, selected_attribute_array, use_case_name,
                                    attributes_string, classification_method, cohort_title, sampling_title, plot_performance_metrics,
                                    save_to_file: False):
    # Use fairlearn MetricFrame to directly plot selected metrics https://fairlearn.org/v0.8/user_guide/assessment/plotting.html
    performance_metrics = {'accuracy': accuracy_score,
                           'recall': recall_score,
                           'precision': precision_score,
                           'auroc score': roc_auc_score,
                           'selection rate': selection_rate,
                           'count': count}

    y_pred_privileged = y_pred[selected_attribute_array[selected_attribute_array == 1]]
    y_true_privileged = y_true[selected_attribute_array[selected_attribute_array == 1].index].to_numpy()
    y_pred_unprivileged = y_pred[selected_attribute_array[selected_attribute_array == 0]]
    y_true_unprivileged = y_true[selected_attribute_array[selected_attribute_array == 0].index].to_numpy()

    # todo future research: clusters are simply too small dummys have too strong influence on them
    ## any other way to calculate recall even if no TP?
    # Problem: Using Dummys does not work. Simply not enough cases if no TP, display warning in frontend
    # Idea: if selection too small, privileged class has no death cases, recall and precision = 0
    ## must be calculated manually and with dummy, otherwise no useful metrics
    ## add dummy values if privileged class has no TP
    ## For loop not ideal when working with arrays, but works
    # maybe better: np.count_nonzero(y_pred_privileged == y_true_privileged)
    true_positives = 0
    for i, real_value in enumerate(y_true_privileged):
        if real_value == 1:
            predicted_value = y_pred_privileged[i]
            if real_value == predicted_value:
                true_positives += 1
    if true_positives == 0:
        # Adding one dummy TP, FP, FN, TN to the predictions for privileged class -> recall and precision can be calculated
        # TP
        selected_attribute_array[-1] = 1
        y_true[-1] = 1
        y_pred = np.append(y_pred, 1)
        # FP
        selected_attribute_array[-2] = 1
        y_true[-2] = 0
        y_pred = np.append(y_pred, 1)
        # FN
        selected_attribute_array[-3] = 1
        y_true[-3] = 1
        y_pred = np.append(y_pred, 0)
        # TN
        selected_attribute_array[-4] = 1
        y_true[-4] = 0
        y_pred = np.append(y_pred, 0)

    try:
        performance_obj = MetricFrame(metrics=performance_metrics,
                                       y_true=y_true.to_numpy(),
                                       y_pred=y_pred,
                                       sensitive_features=selected_attribute_array)
    except ValueError:
        # st.warning('Warning: ValueError occurred, because only one class available for fairness analysis.')
        return None, None, None

    # Customize plots with ylim
    figure_object = performance_obj.by_group.plot(
        kind="bar",
        ylim=[0, 1],
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[12, 8],
        title=f'Metrics per Subgroup on {attributes_string}')
    count_axis = figure_object[1][2]
    y_limit = len(y_pred) + 50
    count_axis.set_ylim(bottom=0, top=y_limit)
    performance_metrics_plot = figure_object[0][0].figure

    # Get metrics_per_group_df
    metrics_per_group_df = performance_obj.by_group
    metrics_per_group_df.loc['overall', ['accuracy', 'recall', 'precision', 'auroc score', 'selection rate', 'count']] = performance_obj.overall
    # cols = metrics_per_group_df.columns.tolist()
    # cols = cols[-1:] + cols[:-1]
    # metrics_per_group_df = metrics_per_group_df[cols]
    metrics_per_group_df.loc[:, ['accuracy','recall',  'precision', 'auroc score', 'selection rate']] = metrics_per_group_df.loc[:,
                                                                        ['accuracy', 'recall', 'precision', 'auroc score',
                                                                         'selection rate']].round(3)
    metrics_per_group_df = metrics_per_group_df.transpose()

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")  # removed %d%m%Y_ from date
        performance_metrics_plot.savefig(f'./output/{use_case_name}/classification/PLOT_FAIRNESS_{attributes_string}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.png',
            dpi=600)
        filename_string: str = f'./output/{use_case_name}/classification/GROUP_FAIRNESS_{attributes_string}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            metrics_per_group_df.to_csv(output_file, index=True)
            print(f'STATUS: metrics_per_group_df was saved to {filename_string}')

    if plot_performance_metrics:
        plt.show()
    else:
        # print('CHECK: Overall metrics:')
        # print(metric_frame.overall)
        print('CHECK: Created metrics_per_group_df.')
        # print(metric_frame.by_group)
        # plt.show()
    # plt.close()

    # New version for fairness metrics report:
    fairness_metrics = {'accuracy': accuracy_score,
                        'recall': recall_score,
                        'precision': precision_score,
                        'selection_rate': selection_rate,
                        'confusion_matrix': confusion_matrix}
    fairness_obj = MetricFrame(metrics=fairness_metrics,
                                  y_true=y_true.to_numpy(),
                                  y_pred=y_pred,
                                  sensitive_features=selected_attribute_array)
    fairness_report = pd.DataFrame(columns=['Values'])

    # Get group_fairness metrics 1.1 (Performance Parity)
    # todo future work: maybe also use ratio (unprivileged/privileged) as a second column in report, next to difference
    group_report = fairness_obj.by_group
    group_report = group_report.transpose()
    temp_differences = group_report.diff(periods=-1, axis=1)
    try:
        group_report['diff'] = temp_differences[0]      # add differences to unprivileged class (0) to group_report
    except KeyError:
        # st.warning('Warning: KeyError occurred, because only one class available for fairness analysis.')
        return None, None, None

    accuracy_parity = group_report.loc['accuracy', 'diff']
    recall_parity = group_report.loc['recall', 'diff']
    precision_parity = group_report.loc['precision', 'diff']
    demographic_parity= group_report.loc['selection_rate', 'diff']

    fairness_report.loc['Accuracy Parity'] = round(accuracy_parity, 3)
    fairness_report.loc['Recall Parity'] = round(recall_parity, 3)
    fairness_report.loc['Precision Parity'] = round(precision_parity, 3)
    fairness_report.loc['Demographic Parity'] = round(demographic_parity, 3)        # = difference in selection rates

    # Confusion Matrix: rows=real, columns=predicted, 0=survived, 1=death
    confusion_matrix_overall = pd.DataFrame(fairness_obj.overall['confusion_matrix'], columns=['predicted_0', 'predicted_1'])
    try:
        confusion_matrix_unprivileged = pd.DataFrame(group_report.loc['confusion_matrix', 0], columns=['predicted_0', 'predicted_1'])
        confusion_matrix_privileged = pd.DataFrame(group_report.loc['confusion_matrix', 1], columns=['predicted_0', 'predicted_1'])
    except KeyError as e:
        # st.warning('Warning: KeyError occurred, because only one class available for fairness analysis.')
        return None, None, None

    tpr_overall, fpr_overall = get_rates_from_cm(confusion_matrix_overall)
    tpr_priv, fpr_priv = get_rates_from_cm(confusion_matrix_privileged)
    tpr_unpriv, fpr_unpriv = get_rates_from_cm(confusion_matrix_unprivileged)

    # Performance Rates for individual metrics (2.1 Equalized Odds and 2.2 Equal Opportunity)
    equalized_odds = (tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv)
    equal_opportunity = tpr_unpriv - tpr_priv
    fairness_report.loc['Equalized Odds'] = round(equalized_odds, 3)
    fairness_report.loc['Equal Opportunity'] = round(equal_opportunity, 3)

    # Get General Fairness Metric 3. (Entropy with Theil Index)
    # todo future research: entropy is calculated per subgroup here, could also be displayed in the
    ## performance metrics table. Actually all fairness metrics could be displayed in there.
    ## maybe better to only have one table for this complete topic.

    entropy_index_unprivileged = generalized_entropy_error(y_true=y_true_unprivileged, y_pred=y_pred_unprivileged, alpha=1, pos_label=1)        # using Theil-Index for Entropy Metric
    entropy_index_privileged = generalized_entropy_error(y_true=y_true_privileged, y_pred=y_pred_privileged, alpha=1, pos_label=1)        # using Theil-Index for Entropy Metric
    entropy_diff = entropy_index_unprivileged - entropy_index_privileged

    # print('CHECK Entropy: ')
    # print(entropy_index_privileged)
    # print(entropy_index_unprivileged)
    fairness_report.loc['Entropy Difference'] = round(entropy_diff, 3)
    fairness_report.index.names = ['Metrics']

    return performance_metrics_plot, metrics_per_group_df, fairness_report


def get_factorized_values(feature, privileged_values, factorization_df):
    factorized_values = []

    temp_factorization_df = factorization_df.loc[factorization_df['feature'] == feature]
    for unfactorized_value in privileged_values:
        temp_fact_value = temp_factorization_df.loc[temp_factorization_df['unfactorized_value'] == unfactorized_value, 'factorized_value'].item()
        factorized_values.append(temp_fact_value)

    return factorized_values


def get_aif360_report(merged_test_data, selected_dependent_variable, selected_protected_attributes,
                      selected_privileged_classes, predicted_labels, attributes_string):
    # Deprecated
    dataset = StandardDataset(df=merged_test_data,
                              label_name=selected_dependent_variable,
                              favorable_classes=[1],  # 'death' has to be used as 'favorable_class'
                              protected_attribute_names=selected_protected_attributes,
                              privileged_classes=selected_privileged_classes)

    dataset_pred = dataset.copy()
    dataset_pred.labels = predicted_labels

    # ERROR of aif360 approach comes from here. For multiple privileged groups (WHITE + MALE) only the intersection (AND) are selected.
    # This is correct for the selection of privileged cases but not for the unprivileged cases.
    # ASIAN + MALE would not be selected for unprivileged cases.
    try:
        privileged_groups = {}
        unprivileged_groups = {}
        for attr in dataset_pred.protected_attribute_names:
            # maybe future research: currently only selection of one value per protected_attribute_names, maybe in future more useful, for this iterate through idx?
            idx = dataset_pred.protected_attribute_names.index(attr)
            privileged_groups[attr] = dataset_pred.privileged_protected_attributes[idx][0]
            unprivileged_groups[attr] = dataset_pred.unprivileged_protected_attributes[idx][0]

        privileged_groups = [privileged_groups]
        unprivileged_groups = [unprivileged_groups]

    except IndexError as e:
        print('Warning: IndexError because no privileged attributes selected.', e)
        return None, None, None, None

    # 3) get metric objs for the report
    classification_metric = ClassificationMetric(dataset=dataset,
                                                 classified_dataset=dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    # 4) Calculate Fairness Metrics
    num_instances_privileged = classification_metric.num_instances(privileged=True)
    num_instances_unprivileged = classification_metric.num_instances(privileged=False)

    accuracy_overall = round(classification_metric.accuracy(), 3)
    accuracy_privileged = round(classification_metric.accuracy(privileged=True), 3)
    accuracy_unprivileged = round(classification_metric.accuracy(privileged=False), 3)
    accuracy = accuracy_unprivileged - accuracy_privileged
    accuracy_def = 'Comparing performance for accuracy'
    accuracy_expected = 0

    recall_overall = round(classification_metric.recall(), 3)
    recall_privileged = round(classification_metric.recall(privileged=True), 3)
    recall_unprivileged = round(classification_metric.recall(privileged=False), 3)
    recall = recall_unprivileged - recall_privileged
    recall_def = 'Comparing performance for Recall (TP/Real Positives)'
    recall_expected = 0

    precision_overall = round(classification_metric.precision(), 3)
    precision_privileged = round(classification_metric.precision(privileged=True), 3)
    precision_unprivileged = round(classification_metric.precision(privileged=False), 3)
    precision = precision_unprivileged - precision_privileged
    precision_def = 'Comparing performance for Precision (TP/Predicted Positives)'
    precision_expected = 0

    num_instances = round(classification_metric.num_instances(), 0)
    num_instances_def = 'Instances used for prediction'
    num_instances_expected = '-'
    statistical_parity_difference = round(classification_metric.statistical_parity_difference(), 3)
    parity_def = 'Alias for Statistical Parity Difference'
    parity_expected = 0
    disparate_impact = round(classification_metric.disparate_impact(), 3)
    disparate_def = 'Alias for Disparate Impact Ratio'
    disparate_expected = 1
    true_positive_rate_difference = round(classification_metric.true_positive_rate_difference(), 3)
    tp_rate_def = 'Alias for True Positive Rate Difference'
    tp_rate_expected = 0
    false_negative_rate_difference = round(classification_metric.false_negative_rate_difference(), 3)
    fp_rate_def = 'False Negative Rate, used for Equalized Odds'
    fp_rate_expected = 0
    average_odds_difference = round(classification_metric.average_odds_difference(), 3)
    average_odds_def = 'Alias for Average Odds'
    average_odds_expected = 0
    generalized_entropy_index = round(classification_metric.generalized_entropy_index(), 3)
    entropy_def = 'Alias for Theil Index with alpha=1. Optimal value=0'
    entropy_expected = 0
    # OPTIONAL metric:
    # differential_fairness_bias_amplification = round(classification_metric.differential_fairness_bias_amplification(), 3)
    # compares empirical_differential_fairness between original and classified dataset
    # edf = occurrence of positive cases between privileged and unprivileged
    # for the question: does my classifier make the occurrence of positive cases higher?

    # 4) return Fairness Report as print if verbose, save as table if save_files
    report = pd.DataFrame({'Number of Instances': [num_instances, num_instances_expected, num_instances_def],
                           # why were these commented out? Not same values as Plot?
                           'num_instances_privileged': num_instances_privileged,
                           'num_instances_unprivileged': num_instances_unprivileged,
                           '0. Accuracy Overall': accuracy_overall,
                           '0. Recall Overall': recall_overall,
                           '0. Precision Overall': precision_overall,
                           '0.1 Accuracy Privileged': accuracy_privileged,
                           '0.1 Recall Privileged': recall_privileged,
                           '0.1 Precision Privileged': precision_privileged,

                           '1.1 Accuracy Parity Difference': [accuracy, accuracy_expected, accuracy_def],
                           '1.1 Recall Parity Difference': [recall, recall_expected, recall_def],
                           '1.1 Precision Parity Difference': [precision, precision_expected, precision_def],
                           '1.2 Demographic Parity Difference': [statistical_parity_difference, parity_expected,
                                                                 parity_def],
                           # '1.2 Disparate Impact Ratio': [disparate_impact, disparate_expected, disparate_def],
                           '2.1 Equalized Odds Difference': [average_odds_difference, average_odds_expected,
                                                             average_odds_def],
                           '2.2 Equal Opportunity Difference': [true_positive_rate_difference, tp_rate_expected,
                                                                tp_rate_def],
                           # 'False Negative Rate Difference': [false_negative_rate_difference, fp_rate_expected, fp_rate_def],
                           # 'differential_fairness_bias_amplification': [differential_fairness_bias_amplification],
                           '3. Generalized Entropy Index': [generalized_entropy_index, entropy_expected, entropy_def]
                           })
    report = report.transpose()
    report.index.names = ['Metrics']
    report.rename(columns={0: attributes_string, 1: 'Optimum', 2: 'Information'}, inplace=True)
    # remove the optimum column, as only 0 values for differences
    report = report.loc[:, [attributes_string, 'Information']]

    return report


# @st.cache_data
def get_fairness_report(use_this_function: False, selected_cohort, cohort_title: str, features_df,
                        selected_features: list, selected_dependent_variable: str, classification_method: str,
                        sampling_method: str, use_case_name, save_to_file, plot_performance_metrics: False,
                        test_size, use_grid_search: False, verbose: True, protected_features, privileged_values):
    # calculate fairness metrics and return fairness-report
    if not use_this_function:
        return None, None, None, None

    # 0) get_classification_basics
    if classification_method == 'deeplearning_sequential':
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
            selected_cohort=selected_cohort, cohort_title=cohort_title, features_df=features_df,
            selected_features=selected_features,
            selected_dependent_variable=selected_dependent_variable, test_size=test_size,
            sampling_method=sampling_method, verbose=verbose)
        model, history = get_sequential_model(x_train_final=x_train_final, y_train_final=y_train_final)
        predicted_labels = model.predict(x=x_test_basic, batch_size=128).round()
        # round() needed to get from sigmoid probability to class value 0 or 1
    else:
        clf, x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data(
            selected_cohort, cohort_title, features_df, selected_features,
            selected_dependent_variable, classification_method, sampling_method, test_size, use_grid_search, verbose)
        predicted_labels = clf.predict(x_test_basic)

    # 1) select unprivileged_groups and their respective values/classes | would have probably been better built with dict structure
    # IMPORTANT: unfactorized values have to be translated into factorized feature column names and value = 1
    factorization_df = pd.read_excel('./supplements/FACTORIZATION_TABLE.xlsx')  # columns: feature	unfactorized_value	factorized_value
    features_to_factorize = pd.unique(factorization_df['feature']).tolist()
    selected_protected_attributes = []
    selected_privileged_classes = []
    features_no_need_title_value = []
    # clean_privileged_values = []
    # for value_with_number in privileged_values:
    #     clean_privileged_values.append(value_with_number[1:])

    for i, feature in enumerate(protected_features):
        if feature == 'gender':
            # do not refactorize for gender and stroke_type, they are not moved into separate columns when doing factorization
            factorized_values = get_factorized_values(feature=feature, privileged_values=privileged_values[i], factorization_df=factorization_df)
            selected_protected_attributes.append(feature)
            selected_privileged_classes.append(factorized_values)

            # invert the original data for gender: both selections possible
            if 0 in factorized_values and 1 in factorized_values:
                x_test_basic.loc[x_test_basic['gender'] == 0, 'gender'] = 1  # set female = 1, makes all = 1
            elif not 0 in factorized_values and 1 in factorized_values:
                pass                                                           # no changes needed
            elif 0 in factorized_values and not 1 in factorized_values:
                x_test_basic['gender'] = x_test_basic['gender'].map(lambda x: 1 if x == 0 else 0)
            else:
                x_test_basic.loc[x_test_basic['gender'] == 1, 'gender'] = 0   # set male = 0, makes all = 0
        elif feature == 'stroke_type':
            factorized_values = get_factorized_values(feature=feature, privileged_values=privileged_values[i],
                                                      factorization_df=factorization_df)
            selected_protected_attributes.append(feature)
            selected_privileged_classes.append(factorized_values)
            # invert the original data for stroke: only one selection possible
            if 0 in factorized_values:
                x_test_basic['stroke_type'] = x_test_basic['stroke_type'].map(lambda x: 1 if x == 0 else 0)
            elif 0.5 in factorized_values:  # for stroke also need to  transform 0.5 into the new '1'
                x_test_basic['stroke_type'] = x_test_basic['stroke_type'].map(lambda x: 1 if x == 0.5 else 0)
            elif 1 in factorized_values:
                x_test_basic['stroke_type'] = x_test_basic['stroke_type'].map(lambda x: 1 if x == 1 else 0)
        elif feature in features_to_factorize and not (feature == 'stroke_type' or feature == 'gender'):
            factorized_values = get_factorized_values(feature=feature, privileged_values=privileged_values[i], factorization_df=factorization_df)
            for value in factorized_values:
                selected_protected_attributes.append(feature + f'_{value}')
                selected_privileged_classes.append([1])
                features_no_need_title_value.append(feature + f'_{value}')
        elif feature == 'cluster':
            selected_protected_attributes.append(feature)
            for value in privileged_values:
                selected_privileged_classes.append(value)
        else:
            selected_protected_attributes.append(feature)
            selected_privileged_classes.append(privileged_values[i])

    # Create attributes_string for title
    attributes_string = ''
    for i, feature in enumerate(selected_protected_attributes):
        if feature in features_no_need_title_value:
            factorized_value = feature[len(feature)-1:]                        # not clean if feature number > 9
            temp_factorization_df = factorization_df.loc[factorization_df['feature'] == feature[:-2]]
            try:
                unfactorized_value = temp_factorization_df.loc[temp_factorization_df['factorized_value'] == int(factorized_value), 'unfactorized_value'].to_list()
                unfactorized_value = unfactorized_value[0]
            except IndexError:
                unfactorized_value = 'unknown'
            attributes_string += '_' + feature[:-1]  + str(unfactorized_value)
        else:
            temp_values = ''
            available_values = selected_privileged_classes[i]
            for value in available_values:
                try:
                    temp_values += str(int(value)) + '_'
                except TypeError:
                    temp_values += str(value) + '_'
            attributes_string += '_' + feature + '_' + temp_values[:-1]
    attributes_string = attributes_string[1:]
    # attributes_string = '_'.join(str(e) for e in selected_protected_attributes)

    # 2) Deprecated: aif360 StandardDataset, selection of unprivileged_groups was not reliable
    # merged_test_data = x_test_basic.merge(right=y_test_basic, left_index=True, right_index=True)
    # old_report = get_aif360_report(merged_test_data, selected_dependent_variable, selected_protected_attributes,
    #                             selected_privileged_classes, predicted_labels, attributes_string)
    # if verbose:
    #     print(f'\n CHECK: AIF360 Fairness Report (not reliable) for {classification_method} on {cohort_title}, {sampling_title}:')
    #     print(old_report.to_string())

    # 2) New: Get Fairness Metrics from Performance Metrics Plot Function
    if classification_method == 'deeplearning_sequential':      # still a problem with layout of predicted_labels, for now not important
        fairness_report = None
        performance_metrics_plot = None
        performance_per_group_df = None
    else:
        if selected_protected_attributes[0] == 'cluster':
            # invert the clusters column from cluster numbers to only '1' at selected cluster
            temp_series = x_test_basic[selected_protected_attributes].squeeze() # should only be one column to series
            temp_df = temp_series.isin(selected_privileged_classes[0]).astype(int).to_frame()
        else:
            temp_df = x_test_basic[selected_protected_attributes]

        # Set 1 where ALL combined selected columns contain a 1
        all_ones_array = temp_df.apply(lambda x: all(x == 1), axis=1).astype(int)
        # print('CHECK all ones array: ')
        # print(all_ones_array)
        performance_metrics_plot, performance_per_group_df, fairness_report = create_performance_metrics_plot(y_pred=predicted_labels,
                                        y_true=y_test_basic,
                                        selected_attribute_array=all_ones_array,
                                        use_case_name=use_case_name,
                                        attributes_string=attributes_string,
                                        classification_method=classification_method,
                                        cohort_title=cohort_title,
                                        sampling_title=sampling_title,
                                        plot_performance_metrics=plot_performance_metrics,
                                        save_to_file=save_to_file)

        if save_to_file:
            current_time = datetime.datetime.now().strftime("%H_%M_%S")  # removed %d%m%Y_ from date
            report_filename_string: str = f'./output/{use_case_name}/classification/FAIRNESS_{attributes_string}_{classification_method}_{cohort_title}_{sampling_title}_{current_time}.csv'
            report_filename = report_filename_string.encode()
            # code to export a df
            with open(report_filename, 'w', newline='') as output_file:
                fairness_report.to_csv(output_file, index=True)  # keep index here for metrics titles
                print(f'STATUS: fairness_report was saved to {report_filename}')

    return fairness_report, performance_metrics_plot, performance_per_group_df, attributes_string


def plot_radar_fairness(categories, list_of_results):
    # Matplot
    # label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(list_of_results[0]))
    # plt.figure()        # figsize=(8, 8))
    # plt.subplot(polar=True)
    # for i, result in enumerate(list_of_results):
    #     plt.plot(label_loc, result, label=f'Result {i}')
    # # plt.title('Fairness Metrics Radar Chart')       # , size=20)
    # lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    # plt.legend()

    # Plotly
    categories = [*categories, categories[0]]
    data = []
    for i, result in enumerate(list_of_results):
        result = [*result, result[0]]
        data.append(go.Scatterpolar(r=result, theta=categories, name=f'Method {i + 1}'))

    if len(list_of_results) > 1:
        show_legend = True
    else:
        show_legend = False

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            # title=go.layout.Title(text='Fairness comparison'),
            polar={'radialaxis': {'visible': True, 'angle': 90}},
            showlegend=show_legend
        )
    )
    pyo.plot(fig, auto_open=False, auto_play=False)

    return fig
