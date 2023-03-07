import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# also a function needed to turn one-hot-encoding back?
def get_one_hot_encoding(selected_cohort, categorical_features):
    for feature in categorical_features:
        encoder = OneHotEncoder()
        onehotarray = encoder.fit_transform(selected_cohort[[feature]]).toarray()
        items = [f'{feature}_{item}' for item in encoder.categories_[0]]
        selected_cohort[items] = onehotarray
        selected_cohort.drop(columns=feature, inplace=True)  # remove original column

    print('CHECK: count of selected_cohort features after encoding:', len(selected_cohort.columns))
    # print('CHECK: selected_cohort features after encoding:', selected_cohort.columns)

    return selected_cohort


# not used anymore
def create_factorization_table(avg_cohort, features_df, cohort_title, features_to_remove):
    # todo future work: check if this automatic factorization was actually a good solution for categorical features?
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


def get_preprocessed_avg_cohort(avg_cohort, features_df, cohort_title):
    # Features that must always be removed
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    for feature in features_to_remove:
        try:
            avg_cohort = avg_cohort.drop(columns=feature)
        except KeyError as e:
            pass

    # Create automated Factorization tables -> removed because now done inside patients.py > get_clean_raw_data
    # create_factorization_table(avg_cohort, features_df, cohort_title, features_to_remove)

    # Select final features in a list, drop features_to_remove
    selected_features = features_df['feature_name'].loc[features_df['selected_for_analysis'] == 'yes'].to_list()
    for feature in features_to_remove:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass

    return avg_cohort[selected_features]  # output: only filtered selected_features inside df
