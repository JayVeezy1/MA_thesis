from pandas.core.interchange import dataframe


def cleanup_avg_df(avg_patient_cohort, selected_features, selected_dependent_variable) -> dataframe:
    # Preprocessing for classification
    avg_df = avg_patient_cohort.copy()
    avg_df = avg_df.drop(columns=['ethnicity', 'insurance'])  # todo: these features can only be used if numeric - any good way to include?
    selected_features_final = selected_features.copy()
    selected_features_final.remove('icustay_id')
    selected_features_final.remove('stroke_type')
    selected_features_final.remove('ethnicity')
    selected_features_final.remove('insurance')
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
    selected_features_final.remove('icustay_id')
    selected_features_final.remove('stroke_type')
    selected_features_final.remove('ethnicity')
    selected_features_final.remove('insurance')
    selected_features_final.append(selected_dependent_variable)

    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_df = avg_df.drop(columns=available_dependent_variables)
    avg_df = avg_df[selected_features_final]
    avg_df = avg_df.fillna(0)

    return avg_df
