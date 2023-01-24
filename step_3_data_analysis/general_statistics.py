import datetime
import warnings

import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.interchange import dataframe


def calculate_deaths_table(selected_patient_cohort, cohort_title, selected_features, save_to_file):
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
        filename_string: str = f'./output/deaths_overview_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            deaths_df.to_csv(output_file)
            print(f'STATUS: deaths_overview_table was saved to {filename_string}')
    else:
        print(deaths_df.to_string())

    return None


def calculate_feature_overview_table(selected_patient_cohort, cohort_title, selected_features, save_to_file):
    feature_categories_table = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')

    # include dependent_variables (they are also used in the separate death_overview)
    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    selected_features.extend(available_dependent_variables)

    # create overview_df
    data = {'Variables': ['Total'], 'Classification': ['Patients'], 'Count': [len(selected_patient_cohort.index)]}
    overview_df: dataframe = pd.DataFrame(data)

    # fill the overview_df
    for feature in selected_features:  # todo: also put features into categories to sort them (general_info, vital_signs, laboratory_results)
        # normal case, no binning needed
        if feature_categories_table['needs_binning'][
            feature_categories_table['feature_name'] == feature].item() == 'False':
            for appearance in pd.unique(selected_patient_cohort[feature]):
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': [appearance],
                                                   'Count': [selected_patient_cohort[feature][
                                                                 selected_patient_cohort[
                                                                     feature] == appearance].count()]})
                overview_df = pd.concat([overview_df, temp_df], ignore_index=True)
        # binning needed for vital signs, etc.
        elif feature_categories_table['needs_binning'][
            feature_categories_table['feature_name'] == feature].item() == 'True':

            try:
                feature_min = int(np.nanmin(selected_patient_cohort[feature].values))
                feature_max = int(np.nanmax(selected_patient_cohort[feature].values))

                feature_appearances_series: Series = selected_patient_cohort[feature].value_counts(bins=[feature_min,
                                                                                                         feature_min + round(
                                                                                                             (
                                                                                                                     feature_max - feature_min) * 1 / 3,
                                                                                                             0),
                                                                                                         feature_min + round(
                                                                                                             (
                                                                                                                     feature_max - feature_min) * 2 / 3,
                                                                                                             0),
                                                                                                         feature_max])
                binning_intervals: list = list(feature_appearances_series.keys())
                binning_counts: list = list(feature_appearances_series.values)

                # add all bins for the features
                for i in range(0, len(binning_intervals)):
                    temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [str(binning_intervals[i])],
                                                       'Count': [binning_counts[i]]})
                    overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

            except ValueError as e:             # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column found with All-NaN entries: {feature}')
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': ['All Entries NaN'],
                                                   'Count': [0]})
                overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

        # known feature that can be removed
        elif feature == 'icustay_id':
            pass
        # other cases, this contains 'unclear' and 'not_for_classification'
        else:
            temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                               'Classification': [f'no category for: {feature}'],
                                               'Count': '-'})
            overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

    # todo: sort variables, also sort for each variable the appearances (first 1, then 0) or alphabetically, also sort the intervals for vital signs somehow

    # todo: somehow add correlation-to-death + p-value (must be after the correlation calculation?)

    # todo: add NaN amount

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/features_overview_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            overview_df.to_csv(output_file)
            print(f'STATUS: features_overview_table was saved to {filename_string}')
    else:
        print(overview_df.to_string())

    return None
