from pandas.core.interchange import dataframe


def cleanup_avg_df(avg_patient_cohort, selected_features, selected_dependent_variable) -> dataframe:
    # Preprocessing for classification
    avg_df = avg_patient_cohort.copy()
    avg_df = avg_df.drop(columns=['ethnicity', 'insurance'])  # todo: these features can only be used if numeric - any good way to include?
    selected_features_final = selected_features.copy()
    try:
        selected_features_final.remove('icustay_id')
    except ValueError as e:
        print('WARNING: icustay_id has already been removed from dataframe.')
    try:
        selected_features_final.remove('dbsource')
    except ValueError as e:
        print('WARNING: dbsource has already been removed from dataframe.')
    try:
        selected_features_final.remove('stroke_type')
    except ValueError as e:
        print('WARNING: stroke_type has already been removed from dataframe.')
    try:
        selected_features_final.remove('infarct_type')
    except ValueError as e:
        print('WARNING: infarct_type has already been removed from dataframe.')
    try:
        selected_features_final.remove('ethnicity')
    except ValueError as e:
        print('WARNING: ethnicity has already been removed from dataframe.')
    try:
        selected_features_final.remove('insurance')
    except ValueError as e:
        print('WARNING: insurance has already been removed from dataframe.')

    selected_features_final.append(selected_dependent_variable)
    avg_df = avg_df[selected_features_final]
    avg_df = avg_df.fillna(0)

    return avg_df


def cleanup_avg_df_for_classification(avg_patient_cohort, selected_features, selected_dependent_variable) -> dataframe:
    # Preprocessing for classification
    # here also the other available_dependent_variables must be removed from df, not so for clustering etc.
    avg_df = avg_patient_cohort.copy()
    avg_df = avg_df.drop(columns=['ethnicity', 'insurance'])  # todo: these features can only be used if numeric - any good way to include?
    selected_features_final = selected_features.copy()
    try:
        selected_features_final.remove('icustay_id')
    except ValueError as e:
        print('WARNING: icustay_id has already been removed from dataframe.')
    try:
        selected_features_final.remove('dbsource')
    except ValueError as e:
        print('WARNING: dbsource has already been removed from dataframe.')
    try:
        selected_features_final.remove('stroke_type')
    except ValueError as e:
        print('WARNING: stroke_type has already been removed from dataframe.')
    try:
        selected_features_final.remove('infarct_type')
    except ValueError as e:
        print('WARNING: infarct_type has already been removed from dataframe.')
    try:
        selected_features_final.remove('ethnicity')
    except ValueError as e:
        print('WARNING: ethnicity has already been removed from dataframe.')
    try:
        selected_features_final.remove('insurance')
    except ValueError as e:
        print('WARNING: insurance has already been removed from dataframe.')

    selected_features_final.append(selected_dependent_variable)
    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_df = avg_df.drop(columns=available_dependent_variables)
    avg_df = avg_df[selected_features_final]
    avg_df = avg_df.fillna(0)

    return avg_df
