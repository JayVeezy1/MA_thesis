import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import stats, chi2_contingency  # also for chi2_contingency
from dython.nominal import associations  # for categorical correlation


def preprocess_for_correlation(selected_cohort, features_df, selected_features: list,
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
    try:
        selected_features.remove('dbsource')
    except ValueError as e:
        pass

    # save dependent_variable in a df, remove from selected_features
    temp_selected_features = selected_features.copy()
    temp_selected_features.remove(selected_dependent_variable)
    print(f'CHECK: {len(temp_selected_features)} features used for Correlation.')
    selected_cohort = selected_cohort[selected_features].fillna(0)

    return selected_cohort, selected_features


def split_cohort_into_feature_types(preprocessed_cohort, features_df, preprocessed_features: list,
                                    selected_dependent_variable: str):
    # split selected_cohort into categorical_cohort, continuous_cohort, binary_cohort
    categorical_features = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    temp_single_features = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'single_value'].to_list()
    categorical_features.extend(temp_single_features)
    continuous_features = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'continuous'].to_list()
    binary_features = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'flag_value'].to_list()

    # icustay_id is also removed here because not in selected_features -> not needed for correlation
    categorical_features = list(set(categorical_features).intersection(preprocessed_features))
    continuous_features = list(set(continuous_features).intersection(preprocessed_features))
    binary_features = list(set(binary_features).intersection(preprocessed_features))
    categorical_cohort = preprocessed_cohort[categorical_features]
    continuous_cohort = preprocessed_cohort[continuous_features]
    binary_cohort = preprocessed_cohort[binary_features]

    # add the temp_dependent_variable to each of the 2 dfs that dont have it
    temp_dependent_variable_df = preprocessed_cohort[selected_dependent_variable]
    continuous_cohort = continuous_cohort.merge(right=temp_dependent_variable_df, left_index=True, right_index=True)
    binary_cohort = binary_cohort.merge(right=temp_dependent_variable_df, left_index=True, right_index=True)

    return continuous_cohort, categorical_cohort, binary_cohort


def get_continuous_corr(continuous_cohort, selected_dependent_variable):
    # Correlation
    continuous_cohort_corr = continuous_cohort.corr(method='pearson').round(2)      # removed: , numeric_only=False
    death_corr = continuous_cohort_corr[
        selected_dependent_variable]  # only return correlation towards selected_dependent_variable
    death_corr.drop(selected_dependent_variable, inplace=True)
    death_corr = death_corr.rename('correlation')

    # Significance
    cleaned_df = continuous_cohort.fillna(0)
    validity_df = pd.DataFrame()
    try:
        for feature in continuous_cohort.columns:
            r_val, p_val = stats.pearsonr(cleaned_df[feature], continuous_cohort[
                selected_dependent_variable])  # r_val is the correlation coefficient, already inside death_corr
            validity_df[feature] = [round(p_val, 3)]
    except ValueError as e:
        print('WARNING: ValueError for r_val and p_val calculation. Cluster probably only one entity.')
        for feature in continuous_cohort.columns:
            validity_df[feature] = [np.nan]
    # p = p_val.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))       # alternative to turn pval into stars *
    validity_df.drop(columns=selected_dependent_variable, inplace=True)
    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    # Return one df with columns correlation | p_value
    return pd.merge(death_corr, validity_df, left_index=True, right_index=True)


def get_categorical_corr(categorical_cohort, selected_dependent_variable):
    # Correlation
    # Convert categorical columns to object columns (needed for associations package)
    categorical_cohort = categorical_cohort.apply(lambda x: x.astype("object") if x.dtype == "category" else x)
    # Estimate and generate Theil's U association plot
    theils_u = associations(categorical_cohort, nom_nom_assoc='theil', plot=False)
    plt.close()
    # theils_u can contain two objects: ['ax'] is a plot, ['corr'] is a correlation matrix
    death_corr = theils_u['corr'][selected_dependent_variable].round(2)
    # death_corr.drop(selected_dependent_variable, inplace=True)          # not drop here, only in other two functions
    death_corr = death_corr.rename('correlation')
    # Significance: Chi-squared test on a contingency table per feature
    validity_df = pd.DataFrame()
    for feature in categorical_cohort.columns:
        contingency_tab = pd.crosstab(categorical_cohort[feature], categorical_cohort[selected_dependent_variable])
        c, p_val, dof, expected = chi2_contingency(contingency_tab)  # c = chi-squared test statistic, not needed
        validity_df[feature] = [round(p_val, 3)]
    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    # Return one df with columns correlation | p_value
    return pd.merge(death_corr, validity_df, left_index=True, right_index=True)


def get_binary_corr(binary_cohort, selected_dependent_variable):
    # Correlation
    # 0) Normal pearson with corr -> can be done for binary, but not for categorical
    binary_corr = binary_cohort.corr(method='pearson').round(2)         # removed: , numeric_only=False
    death_corr = binary_corr[selected_dependent_variable].round(2)
    death_corr.drop(selected_dependent_variable, inplace=True)
    death_corr = death_corr.rename('correlation')

    # Alternatives to calculate binary correlation (phi coefficient):
    # 1) using matthews, needs different data format, should also result in normal pearson
    # binary_corr = matthews_corrcoef(binary_cohort)              # phi-coefficient for binary features with matthews -> needs predicted values vs actual
    # 2) Using association package with pearson (same result as normal pearson)
    # binary_cohort = binary_cohort.apply(lambda x: x.astype("object") if x.dtype == "category" else x)
    # Estimate and generate Theil's U association plot
    # pearsons_phi = associations(binary_cohort, nom_nom_assoc='pearson', plot=False)     # pearsons r value (continuous) = phi value if both are binary
    # plt.close()
    # pearsons_phi can contain two objects: ['ax'] is a plot, ['corr'] is a correlation matrix
    # death_corr = pearsons_phi['corr'][selected_dependent_variable].round(2)  # only return correlation towards selected_dependent_variable
    # 3) Manual calculation wth formula -> for manual check
    # rho = (n11 * n00 - n10 * n01) / sqrt(n1. * n0. * n.0 * n.1)
    # for feature in binary_cohort.columns:
    #       temp_matches = binary_cohort.groupby(['death_in_hosp', feature]).size().to_list()
    #       temp_matches is sorted: 0/0, 0/1, 1/0, 1/1 with death_in_hosp/feature
    #       match_00 = temp_matches[0]
    #       match_01 = temp_matches[1]
    #       match_10 = temp_matches[2]
    #       match_11 = temp_matches[3]
    #       rho = (match_11 * match_00 - match_10 * match_01) / math.sqrt((match_11+match_10) * (match_00+match_01) * (match_10+match_00) * (match_11+match_01))
    #       print(f'\n {feature}')
    #       print(rho)

    # Significance: Chi-squared test on a contingency table per feature
    validity_df = pd.DataFrame()
    for feature in binary_cohort.columns:
        contingency_tab = pd.crosstab(binary_cohort[feature], binary_cohort[selected_dependent_variable])
        c, p_val, dof, expected = chi2_contingency(contingency_tab)  # c = chi-squared test statistic, not needed
        validity_df[feature] = [round(p_val, 3)]
    validity_df.drop(columns=selected_dependent_variable, inplace=True)
    validity_df = validity_df.rename({0: 'p_value'}).transpose()

    # Return one df with columns correlation | p_value
    return pd.merge(death_corr, validity_df, left_index=True, right_index=True)


def get_correlations_to_dependent_var(selected_cohort, features_df, selected_features: list,
                                      selected_dependent_variable: str):
    ## 0) Preprocessing
    preprocessed_cohort, preprocessed_features = preprocess_for_correlation(
        selected_cohort=selected_cohort,
        features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable)

    ## 1) split avg_cohort up into continuous, categorical, flag(binary) features
    continuous_cohort, categorical_cohort, binary_cohort = split_cohort_into_feature_types(
        preprocessed_cohort=preprocessed_cohort,
        features_df=features_df,
        preprocessed_features=preprocessed_features,
        selected_dependent_variable=selected_dependent_variable)

    ## Correlations & Significance Tests
    # todo future work: associations package might actually offer simultaneous correlation of continuous and categorical features -> use only one function?
    # 2.1) Get correlation and validity (significance) for continuous
    # method: simply .corr -> r-value | significance = pearsonr -> p-value
    continuous_corr = get_continuous_corr(continuous_cohort, selected_dependent_variable)
    # 2.2) Get correlation and validity for categorical
    # todo future work: if occurrence of one category is < 5 chi-squared is not possible -> use "Fisher’s exact test"
    # method: Cramer’s V (symmetrical) or Theil’s U (asymmetrical) | significance = Chi-Squared
    categorical_corr = get_categorical_corr(categorical_cohort, selected_dependent_variable)
    # 2.3) Get correlation and validity for binary
    # todo future work: currently chi-squared works fine, but maybe better to use McNemar’s chi-square?
    # method: tetrachoric correlation, works with .corr() | significance = chi-square
    binary_corr = get_binary_corr(binary_cohort, selected_dependent_variable)

    ## 3) Merge correlations and validity dfs again
    correlation_validity_df = pd.concat([continuous_corr, categorical_corr, binary_corr])
    correlation_validity_df.sort_values(by='correlation', ascending=False, inplace=True)

    # Return one df with columns correlation | p_value (still contains selected_dependent_variable)
    return correlation_validity_df


def plot_correlations(use_this_function: False, use_plot_heatmap: False, use_plot_pairplot: False, cohort_title: str,
                      selected_cohort, features_df, selected_features: list,
                      selected_dependent_variable: str, use_case_name: str, save_to_file: bool = False):
    if not use_this_function:
        return None

    # Calculate Correlations
    print('STATUS: Calculating Correlations.')
    # correlation_validity_df columns = correlation | p-value
    correlation_validity_df = get_correlations_to_dependent_var(selected_cohort=selected_cohort,
                                                                selected_features=selected_features,
                                                                features_df=features_df,
                                                                selected_dependent_variable=selected_dependent_variable)

    # Plot of correlation
    correlation_values = correlation_validity_df['correlation'].drop(selected_dependent_variable)
    fig, ax1 = plt.subplots()
    color = plt.get_cmap('RdYlGn_r')
    max_value = correlation_values.max()
    min_value = correlation_values.min()

    # Add p-values to labels
    labels = []
    features = correlation_validity_df.index.tolist()
    features.remove(selected_dependent_variable)
    for feature in features:
        temp_pval = correlation_validity_df.loc[correlation_validity_df.index == feature, 'p_value'].item()
        # p < 0.05 then (*) | p < 0.01 then (**) | p < 0.001 then (***)
        if temp_pval < 0.001:
            labels.append(feature + ' (***)')
        elif temp_pval < 0.01:
            labels.append(feature + ' (**)')
        elif temp_pval < 0.05:
            labels.append(feature + ' (*)')
        else:
            labels.append(feature + ' ()')

    # old version with correlation on x-axis
    # make certain minimum of x_axis is 0.4
    # y_axis = max(max_value, min_value) + 0.05
    # if y_axis < 0.4:
    #     y_axis = 0.4
    # ax1.bar([i for i in range(len(correlation_values))],
    #         [value for value in correlation_values], label=labels,
    #         color=[color((value - min_value) / 0.5) for value in correlation_values])  # (max_value + 0.05 - min_value))
    # ax1.set_xticks([i for i in range(len(correlation_values))])
    # ax1.set_xticklabels(labels)
    # ax1.set_ylim([-y_axis, y_axis])  # correlation over 0.4 is not expected
    # ax1.set_title(f"Correlation to {selected_dependent_variable} for {cohort_title}", wrap=True)
    # plt.xticks(rotation=90)
    # fig.tight_layout()

    # horizontal barplot for correlations on y-axis
    labels.reverse()
    correlation_values = correlation_values.iloc[::-1]
    ax1.barh(y=[i for i, label in enumerate(labels)],
             width=correlation_values,               # reverse correlation_values so very high correlations at top
             color=[color((value - min_value) / 0.5) for value in
                    correlation_values])
    ax1.set_yticks([i for i in range(len(correlation_values))])
    ax1.set_yticklabels(labels)
    ax1.set_title(f'Correlation to {selected_dependent_variable} for {cohort_title}', wrap=True)
    fig.tight_layout()

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/correlation_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png',
            dpi=600)
        plt.show()
    # plt.close()

    if use_plot_heatmap:
        plot_heatmap(cohort_title, selected_cohort, features_df, selected_features, selected_dependent_variable,
                     use_case_name, save_to_file)
    if use_plot_pairplot:
        plot_pairplot(cohort_title, selected_cohort, features_df, selected_features, selected_dependent_variable,
                      selected_patients=100,
                      use_case_name=use_case_name,
                      save_to_file=save_to_file)

    return plt


def plot_heatmap(cohort_title: str, selected_cohort, features_df, selected_features: list,
                 selected_dependent_variable: str, use_case_name: str,
                 save_to_file: bool = False):
    ## 0) Preprocessing
    preprocessed_cohort, preprocessed_features = preprocess_for_correlation(
        selected_cohort=selected_cohort,
        features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable)

    # not ideal solution. But following features might be used for correlation_to_dependent_variable, not for this function
    try:
        preprocessed_features.remove('icustay_id')
    except ValueError as e:
        pass
    try:
        preprocessed_features.remove('stroke_type')
    except ValueError as e:
        pass

    if len(preprocessed_features) > 25:
        print(
            'WARNING: More than 25 features selected for heatmap. Plot result will not be usable. plot_heatmap is terminated.')
        return None
    elif len(preprocessed_features) > 15:
        print('WARNING: More than 15 features selected for heatmap. plot_heatmap might take longer.')
    print('STATUS: Plotting plot_heatmap.')

    # Calculate ALL correlations
    avg_patient_cohort_corr = preprocessed_cohort[preprocessed_features].corr(numeric_only=False)
    avg_df_corr_without_nan = avg_patient_cohort_corr.fillna(0)
    triangle_mask = np.triu(avg_df_corr_without_nan)  # Getting the Upper Triangle of the co-relation matrix as mask

    # heatmap for ALL labels
    fig, ax2 = plt.subplots()
    sns.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
                cmap='bwr', yticklabels=preprocessed_features, xticklabels=preprocessed_features, ax=ax2,
                mask=triangle_mask)
    ax2.set_title(f'Correlations in {cohort_title}', wrap=True)
    fig.tight_layout()

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/heatmap_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png',
            dpi=600)
    plt.show()

    return None


def plot_pairplot(cohort_title: str, selected_cohort, features_df, selected_features: list,
                  selected_dependent_variable: str, use_case_name: str, selected_patients: int,
                  save_to_file: bool = False):
    ## 0) Preprocessing
    preprocessed_cohort, preprocessed_features = preprocess_for_correlation(
        selected_cohort=selected_cohort,
        features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable)

    # not ideal solution. But following features might be used for correlation_to_dependent_variable, not for this function
    try:
        preprocessed_features.remove('icustay_id')
    except ValueError as e:
        pass
    try:
        preprocessed_features.remove('stroke_type')
    except ValueError as e:
        pass

    # Check amount of features
    if len(preprocessed_features) > 20:
        print(
            'WARNING: More than 20 features selected for pairplot. Plot result will not be usable. plot_pairplot is terminated.')
        return None
    elif len(preprocessed_features) > 15:
        print('WARNING: More than 15 features selected for pairplot. plot_pairplot might take longer.')
    print('STATUS: Plotting plot_pairplot.')

    # Get selected_labels_df
    selected_labels_df = preprocessed_cohort.filter(preprocessed_features, axis=1)
    avg_df_small = selected_labels_df.iloc[:selected_patients]  # scatter plot with only 100 patients

    # Pairplot of selected_features
    sns.set_style('darkgrid')
    pairplot = sns.pairplot(avg_df_small, corner=True, kind='scatter', diag_kind='hist')
    sns.set_style('white')

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/correlations/pairplot_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png',
            dpi=600)
    plt.show()

    return pairplot.figure
