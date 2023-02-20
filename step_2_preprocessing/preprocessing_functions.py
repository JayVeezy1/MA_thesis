import pandas as pd


def create_factorization_table(avg_cohort_factorized, features_to_factorize, cohort_title):
    # create 'factorization_table'
    value_combination_dicts: list = []
    for feature in features_to_factorize:
        for factorized_value in avg_cohort_factorized[feature].unique():
            value_combination_dicts.append(
                {'feature': feature,
                 'factorized_values': factorized_value,
                 'unfactorized_value': avg_cohort_factorized[avg_cohort_factorized[feature] == factorized_value][
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

    features_to_factorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_factorize = [x for x in features_to_factorize if
                             x not in features_to_remove]  # drop features_to_remove from factorization
    for feature in features_to_factorize:
        avg_cohort.loc[:, f'{feature}_unfactorized'] = avg_cohort[feature]  # keep old columns without factorization

    # Factorize the columns, save old values with the 'factorization_table' in supplements
    avg_cohort[features_to_factorize] = avg_cohort[features_to_factorize].apply(lambda x: pd.factorize(x)[0])

    # Features that need factorization          # todo check: if factorization is a good solution for categorical features? Does it even make sense to use non-numeric columns for correlation? But still needed for clustering right?
    create_factorization_table(avg_cohort, features_to_factorize, cohort_title)

    # Select final features in a list, drop features_to_remove
    selected_features = features_df['feature_name'].loc[features_df['selected_for_analysis'] == 'yes'].to_list()
    for feature in features_to_remove:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass

    return avg_cohort[selected_features]  # output: only filtered selected_features inside df
