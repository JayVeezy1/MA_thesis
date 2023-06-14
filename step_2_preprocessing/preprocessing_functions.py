import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from objects.patients import Patient


# also a function needed to turn one-hot-encoding back?
def get_one_hot_encoding(selected_cohort, categorical_features):
    for feature in categorical_features:
        encoder = OneHotEncoder()
        onehotarray = encoder.fit_transform(selected_cohort[[feature]]).toarray()
        items = [f'{feature}_{item}' for item in encoder.categories_[0]]
        selected_cohort[items] = onehotarray
        selected_cohort.drop(columns=feature, inplace=True)  # remove original column

    # print('CHECK: count of selected_cohort features after encoding:', len(selected_cohort.columns))
    # print('CHECK: selected_cohort features after encoding:', selected_cohort.columns)

    return selected_cohort


# not used anymore
def create_factorization_table(avg_cohort, features_df, cohort_title, features_to_remove):
    # todo future work: rework, was this automatic factorization with supplements table a good solution for categorical features? Why do some have problems with correlation and bad influence on clustering?
    features_to_factorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_factorize = [x for x in features_to_factorize if
                             x not in features_to_remove]  # drop features_to_remove from factorization

    for feature in features_to_factorize:
        avg_cohort.loc[:, f'{feature}_unfactorized'] = avg_cohort[feature]  # keep old columns without factorization

    # Factorize the columns, save old values with the 'factorization_table' in supplements
    avg_cohort[features_to_factorize] = avg_cohort[features_to_factorize].apply(lambda x: pd.factorize(x)[0])

    # create 'factorization_table'
    value_combination_dicts: list = []
    for feature in features_to_factorize:
        for factorized_value in avg_cohort[feature].unique():
            value_combination_dicts.append(
                {'feature': feature,
                 'factorized_values': factorized_value,
                 'unfactorized_value': avg_cohort[avg_cohort[feature] == factorized_value][
                     f'{feature}_unfactorized'].iloc[0]}
            )
    factorization_table = pd.DataFrame(value_combination_dicts)
    factorization_table.loc[factorization_table['unfactorized_value'].isnull(), 'unfactorized_value'] = 'no_data'

    filename_string: str = f'./supplements/factorization_table_{cohort_title}.csv'
    filename = filename_string.encode()
    with open(filename, 'w', newline='') as output_file:
        factorization_table.to_csv(output_file, index=False)
        print(f'CHECK: factorization_table was updated in {filename_string}')


def get_preprocessed_avg_cohort(avg_cohort, features_df, selected_database, selected_stroke_type):
    # Features that must always be removed
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    for feature in features_to_remove:
        try:
            avg_cohort = avg_cohort.drop(columns=feature)
        except KeyError as e:
            pass

    # Old idea: Create automated Factorization tables -> removed because now done inside patients.py > get_clean_raw_data
    # create_factorization_table(avg_cohort, features_df, cohort_title, features_to_remove)

    # Select final features in a list, drop features_to_remove
    selected_features = features_df['feature_name'].loc[features_df['selected_for_analysis'] == 'yes'].to_list()
    for feature in features_to_remove:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass

    preprocessed_avg_cohort = avg_cohort[selected_features]  # output: only filtered selected_features inside df

    # Filtering Database
    if selected_database == 'metavision':
        filtered_avg_cohort = preprocessed_avg_cohort[preprocessed_avg_cohort['dbsource'] == 1]
    elif selected_database == 'carevue':
        filtered_avg_cohort = preprocessed_avg_cohort[preprocessed_avg_cohort['dbsource'] == -1]
    else:
       filtered_avg_cohort = preprocessed_avg_cohort

    # Filtering Stroke Type
    if selected_stroke_type == 'ischemic':  # in feature_preprocessing = 1, but because scaling = 0
        filtered_avg_cohort = filtered_avg_cohort[filtered_avg_cohort['stroke_type'] == 0]
    elif selected_stroke_type == 'other_stroke':
        filtered_avg_cohort = filtered_avg_cohort[filtered_avg_cohort['stroke_type'] == 0.5]
    elif selected_stroke_type == 'hemorrhagic':
        filtered_avg_cohort = filtered_avg_cohort[filtered_avg_cohort['stroke_type'] == 1]
    else:
        filtered_avg_cohort = filtered_avg_cohort       # for complete, all stroke_type

    return filtered_avg_cohort


def get_all_cohorts(SELECTED_COHORT, FEATURES_DF, SELECTED_DATABASE):
    scaled_complete_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=SELECTED_COHORT,
                                                                      features_df=FEATURES_DF,
                                                                      selected_database=SELECTED_DATABASE,
                                                                      selected_stroke_type='complete')
    scaled_hemorrhage_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=SELECTED_COHORT,
                                                                        features_df=FEATURES_DF,
                                                                        selected_database=SELECTED_DATABASE,
                                                                        selected_stroke_type='hemorrhagic')
    scaled_ischemic_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=SELECTED_COHORT,
                                                                      features_df=FEATURES_DF,
                                                                      selected_database=SELECTED_DATABASE,
                                                                      selected_stroke_type='ischemic')

    ALL_COHORTS_WITH_TITLES: dict = {'scaled_complete_avg_cohort': scaled_complete_cohort_preprocessed,
                                     'scaled_hemorrhage_avg_cohort': scaled_hemorrhage_cohort_preprocessed,
                                     'scaled_ischemic_avg_cohort': scaled_ischemic_cohort_preprocessed}

    return ALL_COHORTS_WITH_TITLES