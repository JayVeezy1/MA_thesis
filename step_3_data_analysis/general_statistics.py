import datetime
import warnings

import numpy as np
import pandas as pd
from numpy import sort
from pandas.core.interchange import dataframe

from objects.patients import Patient
from step_3_data_analysis.correlations import get_correlations_on_cohort


def calculate_deaths_table(selected_patient_cohort, cohort_title, use_case_name, selected_features, save_to_file):
    # include dependent_variables (they are also used in the separate death_overview)
    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    selected_features.extend(available_dependent_variables)

    # create deaths_df
    deaths_df: dataframe = pd.DataFrame(
        columns=['case', 'total', 'death_3_days', 'death_30_days', 'death_180_days', 'death_365_days'])

    # fill the counters = respective cells in final table
    alive_counter = 0
    death_in_hosp_counter = 0
    death_not_in_hosp_counter = 0
    death_3_days_counter_inside = 0
    death_30_days_counter_inside = 0
    death_180_days_counter_inside = 0
    death_360_days_counter_inside = 0
    death_3_days_counter_outside = 0
    death_30_days_counter_outside = 0
    death_180_days_counter_outside = 0
    death_360_days_counter_outside = 0

    for index, row in selected_patient_cohort.iterrows():
        if row['death_in_hosp'] == 1:
            if row['death_3_days'] == 1:
                death_in_hosp_counter += 1
                death_3_days_counter_inside += 1
            elif row['death_30_days'] == 1:
                death_in_hosp_counter += 1
                death_30_days_counter_inside += 1
            elif row['death_180_days'] == 1:
                death_in_hosp_counter += 1
                death_180_days_counter_inside += 1
            elif row['death_365_days'] == 1:
                death_in_hosp_counter += 1
                death_360_days_counter_inside += 1
        elif row['death_in_hosp'] == 0:
            if row['death_3_days'] == 1:
                death_not_in_hosp_counter += 1
                death_3_days_counter_outside += 1
            elif row['death_30_days'] == 1:
                death_not_in_hosp_counter += 1
                death_30_days_counter_outside += 1
            elif row['death_180_days'] == 1:
                death_not_in_hosp_counter += 1
                death_180_days_counter_outside += 1
            elif row['death_365_days'] == 1:
                death_not_in_hosp_counter += 1
                death_360_days_counter_outside += 1
            else:
                alive_counter += 1
        else:
            print('ERROR: Patient death_in_hosp status unknown.')

    # print('CHECK: Sum of found patients:', alive_counter + death_3_days_counter_inside + death_30_days_counter_inside
    #     + death_180_days_counter_inside + death_360_days_counter_inside
    #    + death_3_days_counter_outside + death_30_days_counter_outside
    #   + death_180_days_counter_outside + death_360_days_counter_outside)
    # print('CHECK: total_patients:', len(avg_patient_cohort.index))
    # print('CHECK: alive:', alive_counter)

    deaths_df.loc[len(deaths_df)] = ['death_in_hosp', death_in_hosp_counter, death_3_days_counter_inside,
                                     death_30_days_counter_inside, death_180_days_counter_inside,
                                     death_360_days_counter_inside]
    deaths_df.loc[len(deaths_df)] = ['death_not_in_hosp', death_not_in_hosp_counter, death_3_days_counter_outside,
                                     death_30_days_counter_outside, death_180_days_counter_outside,
                                     death_360_days_counter_outside]
    deaths_df.loc[len(deaths_df)] = ['total_deaths',
                                     death_in_hosp_counter + death_not_in_hosp_counter,
                                     death_3_days_counter_inside + death_3_days_counter_outside,
                                     death_30_days_counter_inside + death_30_days_counter_outside,
                                     death_180_days_counter_inside + death_180_days_counter_outside,
                                     death_360_days_counter_inside + death_360_days_counter_outside]
    deaths_df.loc[len(deaths_df)] = ['total_deaths_perc',
                                     round((death_in_hosp_counter + death_not_in_hosp_counter) / len(
                                         selected_patient_cohort.index), 2),
                                     round((death_3_days_counter_inside + death_3_days_counter_outside) / len(
                                         selected_patient_cohort.index), 2),
                                     round((death_30_days_counter_inside + death_30_days_counter_outside) / len(
                                         selected_patient_cohort.index), 2),
                                     round((death_180_days_counter_inside + death_180_days_counter_outside) / len(
                                         selected_patient_cohort.index), 2),
                                     round((death_360_days_counter_inside + death_360_days_counter_outside) / len(
                                         selected_patient_cohort.index), 2)]
    deaths_df.loc[len(deaths_df)] = ['alive', alive_counter, '-', '-', '-', '-']
    deaths_df.loc[len(deaths_df)] = ['total', len(selected_patient_cohort.index), '-', '-', '-', '-']

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/deaths_overview_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            deaths_df.to_csv(output_file, index=False)
            print(f'STATUS: deaths_overview_table was saved to {filename_string}')
    else:
        print(deaths_df.to_string())

    return None


def calculate_feature_overview_table(selected_patient_cohort, cohort_title, use_case_name, features_df, selected_features, selected_dependent_variable, save_to_file: False):
    # todo maybe: add patients_in_training_set (count/occurrence) and chi-squared-value (for categorical features) to overview_table, and also R-Value from correlation?

    # get correlations per feature
    selected_features_corr = selected_features.copy()
    selected_features_corr.remove('icustay_id')
    selected_features_corr.remove('dbsource')
    selected_features_corr.remove('stroke_type')
    selected_features_corr.remove('infarct_type')
    deaths_correlation_df, p_value, r_value = get_correlations_on_cohort(avg_patient_cohort=selected_patient_cohort.copy(),
                                                                         selected_features_corr=selected_features_corr,
                                                                         selected_dependent_variable=selected_dependent_variable)

    # create overview_df
    data = {'Variables': ['Total'], 'Classification': ['Patients'], 'Count': [len(selected_patient_cohort.index)], 'NaN_Count': ['-'],
            f'Correlation_to_{selected_dependent_variable}': ['-'], 'p_value': ['-']}
    overview_df: dataframe = pd.DataFrame(data)

    # fill the overview_df
    for feature in selected_features_corr:  # todo: also put features into categories to sort them (general_info, vital_signs, laboratory_results)
        # normal case, no binning needed
        if features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'False':
            for appearance in sort(pd.unique(selected_patient_cohort[feature])):
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': [appearance],
                                                   'Count': [selected_patient_cohort[feature][
                                                                 selected_patient_cohort[
                                                                     feature] == appearance].count()],
                                                   'NaN_Count': Patient.get_NAN_for_feature_in_cohort(selected_patient_cohort, feature),
                                                   f'Correlation_to_{selected_dependent_variable}': [deaths_correlation_df[feature].item()],
                                                   'p_value': [p_value[feature].item()]})
                overview_df = pd.concat([overview_df, temp_df], ignore_index=True)
        # binning needed for vital signs, etc.
        elif features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'True':

            try:
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                feature_min = int(np.nanmin(selected_patient_cohort[feature].values))
                feature_max = int(np.nanmax(selected_patient_cohort[feature].values))

                if feature_min == feature_max:
                    feature_appearances_series = selected_patient_cohort[feature].value_counts(bins=[feature_min,
                                                                                                     feature_max])
                else:
                    feature_appearances_series = selected_patient_cohort[feature].value_counts(bins=[feature_min,
                                                                                                             feature_min + round(
                                                                                                                 (
                                                                                                                         feature_max - feature_min) * 1 / 3,
                                                                                                                 0),
                                                                                                             feature_min + round(
                                                                                                                 (
                                                                                                                         feature_max - feature_min) * 2 / 3,
                                                                                                                 0),
                                                                                                             feature_max])
                feature_appearances_df = pd.DataFrame()
                feature_appearances_df['intervals'] = feature_appearances_series.keys()
                feature_appearances_df['counts'] = feature_appearances_series.values
                feature_appearances_df['interval_starts'] = feature_appearances_df['intervals'].map(lambda x: x.left)
                feature_appearances_df = feature_appearances_df.sort_values(by='interval_starts')
                binning_intervals: list = feature_appearances_df['intervals'].to_list()
                binning_counts: list = feature_appearances_df['counts'].to_list()

                # add all bins for the features
                temp_nan: int = Patient.get_NAN_for_feature_in_cohort(selected_patient_cohort, feature)     # NaN is the same for each bin
                for i in range(0, len(binning_intervals)):
                    temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [str(binning_intervals[i])],
                                                       'Count': [binning_counts[i]],
                                                       'NaN_Count': temp_nan,
                                                       f'Correlation_to_{selected_dependent_variable}': [deaths_correlation_df[feature].item()],
                                                       'p_value': [p_value[feature].item()]})
                    overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

            except ValueError as e:             # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column {feature} has Error-Message: {e}')
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': ['All Entries NaN'],
                                                   'Count': [0],
                                                   'NaN_Count': len(selected_patient_cohort),
                                                   f'Correlation_to_{selected_dependent_variable}': ['-'],
                                                   'p_value': ['-']})
                overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

        # known feature that can be removed
        elif feature == 'icustay_id':
            pass
        # other cases, this contains 'unclear' and 'not_for_classification'
        else:
            temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                               'Classification': [f'no category for: {feature}'],
                                               'Count': '-',
                                               'NaN_Count': '-',
                                               f'Correlation_to_{selected_dependent_variable}': ['-'],
                                               'p_value': ['-']})
            overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/features_overview_table_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            overview_df.to_csv(output_file, index=False)
            print(f'STATUS: features_overview_table was saved to {filename_string}')
    else:
        print(overview_df.to_string())

    return None
