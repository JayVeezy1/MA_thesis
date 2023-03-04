import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.interchange import dataframe
from scipy.stats import stats, chi2_contingency  # also for chi2_contingency
from dython.nominal import associations             # for categorical correlation
from sklearn.metrics import matthews_corrcoef


def preprocess_for_correlation(selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                               selected_dependent_variable: str):
    # special preprocessing for correlation: selected_cohort has to be split into 3 categories (continuous, categorical, binary)

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

    # save dependent_variable in a df, remove from selected_features
    temp_selected_features = selected_features.copy()
    temp_dependent_variable_df: dataframe = selected_cohort[selected_dependent_variable]
    temp_selected_features.remove(selected_dependent_variable)
    print(f'CHECK: {len(temp_selected_features)} features used for Correlation.')
    selected_cohort = selected_cohort[selected_features].fillna(0)

    # split selected_cohort into categorical_cohort, continuous_cohort, binary_cohort
    categorical_features = features_df['feature_name'].loc[features_df['categorical_or_continuous'] == 'categorical'].to_list()
    temp_single_features = features_df['feature_name'].loc[features_df['categorical_or_continuous'] == 'single_value'].to_list()
    categorical_features.extend(temp_single_features)
    continuous_features = features_df['feature_name'].loc[features_df['categorical_or_continuous'] == 'continuous'].to_list()
    binary_features = features_df['feature_name'].loc[features_df['categorical_or_continuous'] == 'flag_value'].to_list()

    # icustay_id is also removed here because not in selected_features -> not needed for correlation
    categorical_features = list(set(categorical_features).intersection(selected_features))
    continuous_features = list(set(continuous_features).intersection(selected_features))
    binary_features = list(set(binary_features).intersection(selected_features))
    categorical_cohort = selected_cohort[categorical_features]
    continuous_cohort = selected_cohort[continuous_features]
    binary_cohort = selected_cohort[binary_features]

    # add the temp_dependent_variable to each of the 3 dfs
    # categorical_cohort = categorical_cohort.merge(right=temp_dependent_variable_df, left_index=True, right_index=True) # not needed, dependent_variable is categorical
    continuous_cohort = continuous_cohort.merge(right=temp_dependent_variable_df, left_index=True, right_index=True)
    binary_cohort = binary_cohort.merge(right=temp_dependent_variable_df, left_index=True, right_index=True)

    return continuous_cohort, categorical_cohort, binary_cohort, selected_features          # selected_features might not be needed


def get_continuous_corr(continuous_cohort, selected_dependent_variable):
    # Correlation
    continuous_cohort_corr = continuous_cohort.corr(method='pearson', numeric_only=False).round(2)
    death_corr = continuous_cohort_corr[selected_dependent_variable]  # only return correlation towards selected_dependent_variable
    # death_corr.drop(selected_dependent_variable, inplace=True)      # not drop here, only in other two functions

    # Significance
    cleaned_df: dataframe = continuous_cohort.fillna(0)
    validity_df: dataframe = pd.DataFrame()
    try:
        for feature in continuous_cohort.columns:
            r_val, p_val = stats.pearsonr(cleaned_df[feature], continuous_cohort[selected_dependent_variable])  # r_val is the correlation coefficient, already inside death_corr
            validity_df[feature] = [round(p_val, 3)]
    except ValueError as e:
        print('WARNING: ValueError for r_val and p_val calculation. Cluster probably only one entity.')
        for feature in continuous_cohort.columns:
            validity_df[feature] = [np.nan]
    # p = p_val.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))       # alternative to turn pval into stars *

    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    return death_corr, validity_df


def get_categorical_corr(categorical_cohort, selected_dependent_variable):
    # Correlation
    # Convert categorical columns to object columns (needed for associations package)
    categorical_cohort = categorical_cohort.apply(lambda x: x.astype("object") if x.dtype == "category" else x)
    # Estimate and generate Theil's U association plot
    theils_u = associations(categorical_cohort, nom_nom_assoc='theil', plot=False)
    plt.close()
    # theils_u can contain two objects: ['ax'] is a plot, ['corr'] is a correlation matrix
    death_corr = theils_u['corr'][selected_dependent_variable].round(2)  # only return correlation towards selected_dependent_variable
    death_corr.drop(selected_dependent_variable, inplace=True)

    # Significance: Chi-squared test on a contingency table per feature
    validity_df: dataframe = pd.DataFrame()
    for feature in categorical_cohort.columns:
        contingency_tab = pd.crosstab(categorical_cohort[feature], categorical_cohort[selected_dependent_variable])
        c, p_val, dof, expected = chi2_contingency(contingency_tab)   # c = chi-squared test statistic, not needed
        validity_df[feature] = [round(p_val, 3)]
    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    return death_corr, validity_df


def get_binary_corr(binary_cohort, selected_dependent_variable):
    # Correlation


    # using normal pearson -> values seem wrong
    binary_corr = binary_cohort.corr(method='pearson', numeric_only=False).round(2)
    death_corr = binary_corr[selected_dependent_variable].round(2)
    death_corr.drop(selected_dependent_variable, inplace=True)

    # using matthews, needs different data format, should result in normal pearson
    # binary_corr = matthews_corrcoef(binary_cohort)              # phi-coefficient for binary features with matthews -> needs predicted values vs actual

    # using association package with pearson (same as normal pearson)
    # binary_cohort = binary_cohort.apply(lambda x: x.astype("object") if x.dtype == "category" else x)
    # Estimate and generate Theil's U association plot
    # pearsons_phi = associations(binary_cohort, nom_nom_assoc='pearson', plot=False)     # pearsons r value (continuous) = phi value if both are binary
    # plt.close()
    # pearsons_phi can contain two objects: ['ax'] is a plot, ['corr'] is a correlation matrix
    # death_corr = pearsons_phi['corr'][selected_dependent_variable].round(2)  # only return correlation towards selected_dependent_variable


    # TODO NOW: Calculate binary coefficients 'manually' with formula -> check difference between mechvent and cancer, cancer should be much higher
    # phi coefficient as formula
    # rho = (n11 * n00 - n10 * n01) / sqrt(n11.n10.n01.n00)
    # where    n11(n00) = number    of     rows    with x=1(0) and y=1(0) etc.


    # Significance: Chi-squared test on a contingency table per feature
    validity_df: dataframe = pd.DataFrame()
    for feature in binary_cohort.columns:
        contingency_tab = pd.crosstab(binary_cohort[feature], binary_cohort[selected_dependent_variable])
        c, p_val, dof, expected = chi2_contingency(contingency_tab)  # c = chi-squared test statistic, not needed
        validity_df[feature] = [round(p_val, 3)]
    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    return death_corr, validity_df


def get_correlations_on_cohort(selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                               selected_dependent_variable: str) -> dataframe:
    # todo after: check if this change of 'preprocess_for_correlation' has to be adapted to in other functions

    ## 1 ) split avg_cohort up into continuous, categorical, flag(binary) features
    continuous_cohort, categorical_cohort, binary_cohort, selected_features = preprocess_for_correlation(
                                                                selected_cohort=selected_cohort,
                                                                features_df=features_df,
                                                                selected_features=selected_features,
                                                                selected_dependent_variable=selected_dependent_variable)

    ## Correlations & Significance Tests
    # todo future research: associations package might actually offer simultaneous correlation of continuous and categorical features -> use only one function?
    # 2.1) Get correlation and validity (significance) for continuous
    # correlation = simply .corr -> r-value | significance = pearsonr -> p-value
    continuous_corr, continuous_validity = get_continuous_corr(continuous_cohort, selected_dependent_variable)

    # 2.2) Get correlation and validity for categorical
    # correlation = Cramer’s V (symmetrical) or Theil’s U (asymmetrical) | significance = Chi-Squared but if occurrence of one category is < 5 chi-squared is not possible -> use "Fisher’s exact test"
    categorical_corr, categorical_validity = get_categorical_corr(categorical_cohort, selected_dependent_variable)

    # 2.3) Get correlation and validity for binary
    # correlation = matthews_corrcoef() (tetrachoric correlation) | significance = McNemar’s chi-square
    # TODO NOW: check if theils U and normal chi-squared work as well (like categorical)
    binary_corr, binary_validity = get_binary_corr(binary_cohort, selected_dependent_variable)

    ## TODO: 3) Merge correlations and validity dfs again
    death_corr = pd.concat([continuous_corr, categorical_corr, binary_corr])
    validity_df = pd.concat([continuous_validity, categorical_validity, binary_validity])

    # merge is wrong? Concat better?
    # temp_death_corr = pd.merge(left=continuous_corr, right=categorical_corr, left_index=True, right_index=True)
    # death_corr = pd.merge(left=temp_death_corr, right=binary_corr, left_index=True, right_index=True)
    # temp_validity_df = pd.merge(left=continuous_validity, right=categorical_validity, left_index=True, right_index=True)
    # validity_df = pd.merge(left=temp_validity_df, right=binary_validity, left_index=True, right_index=True)

    return death_corr.sort_values(ascending=False), validity_df


def plot_correlations(use_this_function: False, use_plot_heatmap: False, use_plot_pairplot: False, cohort_title: str,
                      selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                      selected_dependent_variable: str, use_case_name: str, save_to_file: bool = False):
    if not use_this_function:
        return None

    # Calculate Correlations
    print('STATUS: Calculating Correlations.')
    sorted_death_corr, p_value = get_correlations_on_cohort(selected_cohort=selected_cohort,
                                                                     selected_features=selected_features,
                                                                     features_df=features_df,
                                                                     selected_dependent_variable=selected_dependent_variable)

    # todo maybe: also add pval to correlation plot?

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
        plot_heatmap(cohort_title, selected_cohort, features_df, selected_features,
                     selected_dependent_variable, use_case_name,
                     save_to_file)
    if use_plot_pairplot:
        if len(selected_features) > 20:
            print(
                f'WARNING: {len(selected_features)} selected_features were selected for plot_pairplot(). Calculation might take a moment.')
        plot_pairplot(cohort_title, selected_cohort, features_df, selected_features,
                      selected_dependent_variable, use_case_name,
                      save_to_file)

    return None


def plot_heatmap(cohort_title: str, selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                 selected_dependent_variable: str, use_case_name: str,
                 save_to_file: bool = False):
    print('STATUS: Plotting plot_heatmap.')

    # Preprocessing
    avg_cohort, selected_features = preprocess_for_correlation(selected_cohort=selected_cohort,
                                                               features_df=features_df,
                                                               selected_features=selected_features,
                                                               selected_dependent_variable=selected_dependent_variable)
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


def plot_pairplot(cohort_title: str, selected_cohort: dataframe, features_df: dataframe, selected_features: list,
                  selected_dependent_variable: str, use_case_name: str,
                  save_to_file: bool = False):
    print('STATUS: Plotting plot_pairplot.')

    # Preprocessing
    avg_cohort, selected_features = preprocess_for_correlation(selected_cohort=selected_cohort,
                                                               features_df=features_df,
                                                               selected_features=selected_features,
                                                               selected_dependent_variable=selected_dependent_variable)
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
