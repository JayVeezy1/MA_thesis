import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.interchange import dataframe
from scipy.stats import stats


def preprocess_for_correlation(avg_cohort: dataframe, features_df: dataframe, selected_features: list,
                               selected_dependent_variable: str):
    # Preprocessing for Correlation: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    temp_selected_features = selected_features.copy()
    temp_selected_features.remove(selected_dependent_variable)
    print(
        f'CHECK: {len(temp_selected_features)} features used for Correlation.')  # dependent_variable is not used for its own correlation

    avg_cohort = avg_cohort[selected_features]

    return avg_cohort.fillna(0), selected_features


def get_correlations_on_cohort(avg_cohort: dataframe, features_df: dataframe, selected_features: list,
                               selected_dependent_variable: str) -> dataframe:
    avg_cohort, selected_features = preprocess_for_correlation(avg_cohort, selected_features, features_df,
                                                               selected_dependent_variable)

    # Calculate correlation
    avg_patient_cohort_corr = avg_cohort[selected_features].corr(numeric_only=False)
    death_corr = avg_patient_cohort_corr[selected_dependent_variable].round(2)  # only return correlation towards selected_dependent_variable

    # Calculate r-values & p-values             # todo check: Are these calculations correct?
    # old calculation, unclear results
    # pval = avg_patient_cohort[selected_features_corr].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*avg_patient_cohort_corr.shape)
    # p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))       # alternative to turn pval into stars *

    cleaned_df: dataframe = avg_cohort.fillna(value=0)
    validity_df: dataframe = pd.DataFrame()
    try:
        for feature in avg_cohort[selected_features]:
            r_val, p_val = stats.pearsonr(cleaned_df[feature], avg_cohort[selected_dependent_variable])
            validity_df[feature] = [round(r_val, 3), round(p_val, 3)]
    except ValueError as e:
        print('WARNING: ValueError for r_val and p_val calculation. Cluster probably only one entity.')
        for feature in avg_cohort[selected_features]:
            validity_df[feature] = [np.nan, np.nan]

    validity_df = validity_df.rename({0: 'r_value', 1: 'p_value'}).transpose()

    return death_corr.sort_values(ascending=False), validity_df['p_value'], validity_df['r_value']


def plot_correlations(use_this_function: False, use_plot_heatmap: False, use_plot_pairplot: False, cohort_title: str,
                      avg_cohort: dataframe, features_df: dataframe, selected_features: list,
                      selected_dependent_variable: str, use_case_name: str, save_to_file: bool = False):
    if not use_this_function:
        return None

    # Calculate Correlations
    print('STATUS: Calculating Correlations.')
    sorted_death_corr, p_value, r_value = get_correlations_on_cohort(avg_cohort, selected_features, features_df, selected_dependent_variable)

    # todo: also add pval to plot?

    # Plot of correlation
    plot_death_corr = sorted_death_corr.drop(selected_dependent_variable)
    fig, ax1 = plt.subplots()
    color = plt.get_cmap('RdYlGn_r')
    max_value = plot_death_corr.max()
    min_value = plot_death_corr.min()

    # make certain minimum of x_axis is 0.4
    y_axis = max(max_value, min_value) + 0.05
    if y_axis < 0.4:
        y_axis = 0.4
    ax1.bar([i for i in range(len(plot_death_corr))],
            [value for value in plot_death_corr],
            label=plot_death_corr.index.tolist(),
            color=[color((value - min_value) / 0.5) for value in plot_death_corr])  # (max_value + 0.05 - min_value))
    ax1.set_xticks([i for i in range(len(plot_death_corr))])
    ax1.set_xticklabels(plot_death_corr.index.tolist())
    ax1.set_ylim([-y_axis, y_axis])  # correlation over 0.4 is not expected
    ax1.set_title(f"Correlation to {selected_dependent_variable} for {cohort_title}")
    plt.xticks(rotation=90)
    fig.tight_layout()
    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/correlation_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    if use_plot_heatmap:
        plot_heatmap(cohort_title, avg_cohort, features_df, selected_features,
                     selected_dependent_variable, use_case_name,
                     save_to_file)
    if use_plot_pairplot:
        if len(selected_features) > 20:
            print(
                f'WARNING: {len(selected_features)} selected_features were selected for plot_pairplot(). Calculation might take a moment.')
        plot_pairplot(cohort_title, avg_cohort, features_df, selected_features,
                      selected_dependent_variable, use_case_name,
                      save_to_file)

    return None


def plot_heatmap(cohort_title: str, avg_cohort: dataframe, features_df: dataframe, selected_features: list,
                 selected_dependent_variable: str, use_case_name: str,
                 save_to_file: bool = False):
    print('STATUS: Plotting plot_heatmap.')

    # Preprocessing
    avg_cohort, selected_features = preprocess_for_correlation(avg_cohort, features_df, selected_features,
                                                               selected_dependent_variable)
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    # Calculate ALL correlations
    avg_patient_cohort_corr = avg_cohort[selected_features].corr(numeric_only=False)
    avg_df_corr_without_nan = avg_patient_cohort_corr.fillna(0)
    triangle_mask = np.triu(avg_df_corr_without_nan)  # Getting the Upper Triangle of the co-relation matrix as mask

    # heatmap for ALL labels
    fig, ax2 = plt.subplots()
    sns.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
                cmap='bwr', yticklabels=selected_features, xticklabels=selected_features, ax=ax2,
                mask=triangle_mask)
    ax2.set_title(f'Correlations in {cohort_title}')
    fig.tight_layout()

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/heatmap_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    return None


def plot_pairplot(cohort_title: str, avg_cohort: dataframe, features_df: dataframe, selected_features: list,
                  selected_dependent_variable: str, use_case_name: str,
                  save_to_file: bool = False):
    print('STATUS: Plotting plot_pairplot.')

    # Preprocessing
    avg_cohort, selected_features = preprocess_for_correlation(avg_cohort, features_df, selected_features,
                                                               selected_dependent_variable)
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    # Get selected_labels_df
    selected_labels_df = avg_cohort.filter(selected_features, axis=1)
    avg_df_small = selected_labels_df.iloc[:100]  # scatter plot nur 100 patients

    # Pairplot of selected_features
    sns.set_style('darkgrid')
    pairplot = sns.pairplot(avg_df_small, corner=True, kind='scatter', diag_kind='hist')

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/pairplot_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    return None
