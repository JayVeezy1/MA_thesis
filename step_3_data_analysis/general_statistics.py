import pandas as pd
from pandas.core.interchange import dataframe

from objects.patients import Patient


def calculate_deaths_table(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable):
    return None


def calculate_feature_overview_table(avg_patient_cohort, cohort_title, selected_features):
    feature_categories_table = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')

    # include dependent_variables -> also available in separate death_overview
    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    selected_features.extend(available_dependent_variables)

    # create overview_df
    data = {'Variables': ['Total'],
            'Classification': ['Patients'],
            'Count': [len(avg_patient_cohort.index)]}
    overview_df: dataframe = pd.DataFrame(data)

    # fill overview_df
    for feature in selected_features:
        if feature_categories_table['needs_binning'][feature_categories_table['feature_name'] == feature].item() == 'not_for_classification':
            pass
        elif feature_categories_table['needs_binning'][feature_categories_table['feature_name'] == feature].item() == 'unclear':
            temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                               'Classification': [f'no category for: {feature}'],
                                               'Count': '-'})
            overview_df = pd.concat([overview_df, temp_df], ignore_index=True)
        elif feature_categories_table['needs_binning'][feature_categories_table['feature_name'] == feature].item() == 'True':

            # todo: carry on here: vital signs etc need binning for their categories
            temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                               'Classification': [f'needs binning for: {feature}'],
                                               'Count': '-'})
            overview_df = pd.concat([overview_df, temp_df], ignore_index=True)
        else:
            for appearance in pd.unique(avg_patient_cohort[feature]):
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': [appearance],
                                                   'Count': [avg_patient_cohort[feature][avg_patient_cohort[feature] == appearance].count()]})
                # print(temp_df.to_string())
                overview_df = pd.concat([overview_df, temp_df], ignore_index=True)

    # todo: sort for each variable the appearances either 1, then 0 or alphabetically


    # todo: save this general_statistics_table (with cohort name) in output folder -> usable for latex

    print(overview_df.to_string())

    return None
