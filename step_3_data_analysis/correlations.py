import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.interchange import dataframe
from scipy.stats import stats


def get_correlations_on_cohort(avg_patient_cohort, selected_features_corr, selected_dependent_variable) -> dataframe:
    # Preprocessing for correlation
    avg_patient_cohort = avg_patient_cohort.drop(columns=['icustay_id', 'stroke_type', 'infarct_type', 'dbsource'])   # these features are needed for clustering, not correlations

    # todo: make drop and factorize dependent on another column in the feature_preprocessing table

    # todo: check/ask if there is better option? How to use non-numeric columns for correlation? Not really useful? But still needed for clustering right?
    avg_patient_cohort[['insurance', 'ethnicity']] = avg_patient_cohort[['insurance', 'ethnicity']].apply(lambda x: pd.factorize(x)[0])
    # insurance: medicare = 0, mediaid = 1, Government = 2, private = 3, Self Pay = 4
    # ethnicity: WHITE = 0, UNKNOWN/NOT SPECIFIED = 1, HISPANIC OR LATINO = 2, BLACK = 3, OTHER = 4, ASIAN = 5

    # Calculate correlation
    selected_features_corr.append(selected_dependent_variable)
    avg_patient_cohort_corr = avg_patient_cohort[selected_features_corr].corr(numeric_only=False)
    death_corr = avg_patient_cohort_corr[selected_dependent_variable].round(2)     # only return correlation towards selected_dependent_variable

    # Calculate r-values & p-values
    # todo: Are these calculations correct?
    # pval = avg_patient_cohort[selected_features_corr].corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*avg_patient_cohort_corr.shape)        #  old calculation, unclear results
    # p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))       # alternative to turn pval into stars *
    cleaned_df: dataframe = avg_patient_cohort.fillna(value=0)      # filling nan with 0 correct?
    validity_df: dataframe = pd.DataFrame()
    try:
        for feature in avg_patient_cohort[selected_features_corr]:
            r_val, p_val = stats.pearsonr(cleaned_df[feature], avg_patient_cohort[selected_dependent_variable])
            validity_df[feature] = [round(r_val, 3), round(p_val, 3)]
    except ValueError as e:
        print('WARNING: ValueError for r_val and p_val calculation. Cluster probably only one entity.')
        for feature in avg_patient_cohort[selected_features_corr]:
            validity_df[feature] = [np.nan, np.nan]

    validity_df = validity_df.rename({0: 'r_value', 1: 'p_value'}).transpose()

    return death_corr.sort_values(ascending=False), validity_df['p_value'], validity_df['r_value']


def plot_correlations(avg_patient_cohort, cohort_title, use_case_name, selected_features, selected_dependent_variable, save_to_file):
    selected_features_corr = selected_features.copy()
    selected_features_corr.remove('icustay_id')
    selected_features_corr.remove('dbsource')
    selected_features_corr.remove('stroke_type')
    selected_features_corr.remove('infarct_type')

    sorted_death_corr, p_value, r_value = get_correlations_on_cohort(avg_patient_cohort, selected_features_corr, selected_dependent_variable)
    # todo: also add pval to plot?

    # Plot of correlation
    plot_death_corr = sorted_death_corr.drop(selected_dependent_variable)
    fig, ax1 = plt.subplots()
    color = plt.get_cmap('RdYlGn_r')
    max_value = plot_death_corr.max()
    min_value = plot_death_corr.min()
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

    print('STATUS: calculate_correlations_on_cohort executed.')
    return None


def plot_heatmap(avg_patient_cohort, cohort_title, use_case_name, selected_features, selected_dependent_variable, save_to_file):
    # Preprocessing (not inside the normal correlation function, because here ALL correlations needed, not just death)
    selected_features_corr = selected_features.copy()
    selected_features_corr.remove('icustay_id')
    selected_features_corr.remove('stroke_type')
    selected_features_corr.append(selected_dependent_variable)

    # Calculate ALL correlations
    avg_patient_cohort_corr = avg_patient_cohort[selected_features_corr].corr(numeric_only=False)
    avg_df_corr_without_nan = avg_patient_cohort_corr.fillna(0)
    triangle_mask = np.triu(avg_df_corr_without_nan)       # Getting the Upper Triangle of the co-relation matrix to use as mask

    # heatmap for ALL labels
    fig, ax2 = plt.subplots()
    sns.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
                cmap='bwr', yticklabels=selected_features_corr, xticklabels=selected_features_corr, ax=ax2, mask=triangle_mask)
    ax2.set_title(f'Correlations in {cohort_title}')
    fig.tight_layout()

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/heatmap_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    print('STATUS: plot_heatmap executed.')
    return None


def plot_pairplot(avg_patient_cohort, cohort_title, use_case_name, selected_features, selected_dependent_variable, save_to_file):
    selected_features_pair = selected_features.copy()
    selected_features_pair.remove('icustay_id')
    selected_features_pair.remove('stroke_type')
    selected_features_pair.append(selected_dependent_variable)

    selected_labels_df = avg_patient_cohort.filter(selected_features_pair, axis=1)
    avg_df_small = selected_labels_df.iloc[:150]  # scatter plot nur 100 patients

    # Pairplot of selected_features
    sns.set_style('darkgrid')
    pairplot = sns.pairplot(avg_df_small, corner=True, kind='scatter', diag_kind='hist')

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/pairplot_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    print('STATUS: plot_pairplot executed.')
    return None
