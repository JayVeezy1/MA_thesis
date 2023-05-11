import datetime

import pandas as pd
import streamlit as st

from step_3_data_analysis.clustering import get_clusters_overview_table, get_kmeans_clusters, get_overview_for_cluster, \
    plot_sh_score
from step_4_classification.classification import get_auc_score, get_confusion_matrix, get_accuracy, get_recall, get_precision


@st.cache_data
def calculate_clusters_overview_table(use_this_function: False, selected_cohort, cohort_title, use_case_name,
                                      features_df, selected_features, selected_dependent_variable,
                                      selected_k_means_count, show_value_influences, use_encoding: False, save_to_file: False):
    # currently this function is only usable for manually selected cluster_count -> kmeans but not DBSCAN
    if not use_this_function:
        return None

    print('STATUS: Starting with subgroup analysis, cluster comparison.')
    # step 1: get counts for complete dataset -> based on general_statistics.calculate_feature_overview_table
    clusters_overview_table = get_clusters_overview_table(original_cohort=selected_cohort,
                                                          selected_features=selected_features,
                                                          features_df=features_df,
                                                          cohort_title=cohort_title)

    # TODO: make selectable for other clustering methods
    # step 2: get all clusters as df in a list
    kmeans_clusters: list = get_kmeans_clusters(
        original_cohort=selected_cohort,
        features_df=features_df,
        selected_features=selected_features,
        selected_dependent_variable=selected_dependent_variable,
        selected_k_means_count=selected_k_means_count,
        use_encoding=use_encoding,
        verbose=True)

    for i, cluster in enumerate(kmeans_clusters):
        # step 3: get count of occurrences per bin for this cluster
        clusters_overview_table = get_overview_for_cluster(cluster_cohort=cluster,
                                                           original_cohort=selected_cohort,
                                                           selected_features=selected_features,
                                                           features_df=features_df,
                                                           current_overview_table=clusters_overview_table,
                                                           selected_cluster_number=i,
                                                           show_value_influences=show_value_influences)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/subgroup_analysis/clusters_overview_table_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            clusters_overview_table.to_csv(output_file, index=False)
            print(f'STATUS: clusters_overview_table was saved to {filename_string}')
    else:
        print('CHECK: clusters_overview_table:')
        print(clusters_overview_table)

    return clusters_overview_table


@st.cache_data
def compare_classification_models_on_clusters(use_this_function, use_case_name, features_df, selected_features,
                                              selected_cohort, classification_method, sampling_method, cohort_title, clustering_method, dependent_variable,
                                              selected_k_means_count, check_sh_score, use_grid_search: False,
                                              use_encoding: False,
                                              save_to_file):
    # calculate prediction quality per cluster, save into table classification_clusters_overview
    # todo long term: this is only for kmeans, also add for other clustering methods (dynamic like DBSCAN and ASDF also possible?)
    if not use_this_function:
        return None

    if check_sh_score:
        # any option to automatically get optimal cluster count from this?
        plot_sh_score(use_this_function=True,  # True | False
                      selected_cohort=selected_cohort,
                      cohort_title=cohort_title,
                      use_case_name=use_case_name,
                      features_df=features_df,
                      selected_features=selected_features,
                      selected_dependent_variable=dependent_variable,
                      clustering_method=clustering_method,
                      use_encoding=use_encoding,
                      save_to_file=False)

    classification_clusters_overview = pd.DataFrame()


    kmeans_clusters: list = get_kmeans_clusters(original_cohort=selected_cohort,
                                                features_df=features_df,
                                                selected_features=selected_features,
                                                selected_dependent_variable=dependent_variable,
                                                selected_k_means_count=selected_k_means_count,
                                                use_encoding=use_encoding,
                                                verbose=False)

    # get total_auc_score for total set
    total_auc_score, auc_prc_score = get_auc_score(use_this_function=True,  # True | False
                                                   classification_method=classification_method,
                                                   sampling_method=sampling_method,  # SELECTED_SAMPLING_METHOD
                                                   selected_cohort=selected_cohort,
                                                   cohort_title=cohort_title,
                                                   use_case_name=use_case_name,
                                                   features_df=features_df,
                                                   selected_features=selected_features,
                                                   selected_dependent_variable=dependent_variable,
                                                   show_plot=False,
                                                   verbose=False,
                                                   use_grid_search=use_grid_search,
                                                   save_to_file=False)

    total_cm_df = get_confusion_matrix(use_this_function=True,  # True | False
                                       classification_method=classification_method,
                                       sampling_method=sampling_method,  # SELECTED_SAMPLING_METHOD
                                       selected_cohort=selected_cohort,
                                       cohort_title=cohort_title,
                                       use_case_name=use_case_name,
                                       features_df=features_df,
                                       selected_features=selected_features,
                                       selected_dependent_variable=dependent_variable,
                                       use_grid_search=use_grid_search,
                                       verbose=False,
                                       save_to_file=False)

    current_settings = pd.DataFrame([{'dependent_variable': dependent_variable,
                                      'classification_method': classification_method,
                                      'cluster': 'complete_set',
                                      'auc_score': total_auc_score,
                                      'auc_prc_score': auc_prc_score,
                                      'accuracy': get_accuracy(total_cm_df),
                                      'recall': get_recall(total_cm_df),
                                      'precision': get_precision(total_cm_df)
                                      }])
    classification_clusters_overview = pd.concat([classification_clusters_overview, current_settings],
                                                 ignore_index=True)

    for i, cluster in enumerate(kmeans_clusters):               # for each cluster get prediction quality
        print(f'STATUS: Calculating auc_score for cluster: {i}, with model settings: {classification_method}, {dependent_variable}')
        # get auc_score for cluster
        auc_score, auc_prc_score = get_auc_score(use_this_function=True,  # True | False
                                                 classification_method=classification_method,
                                                 sampling_method=sampling_method,
                                                 selected_cohort=cluster,
                                                 cohort_title=f'cluster_{i}',
                                                 use_case_name=use_case_name,
                                                 features_df=features_df,
                                                 selected_features=selected_features,
                                                 selected_dependent_variable=dependent_variable,
                                                 show_plot=False,
                                                 use_grid_search=use_grid_search,
                                                 verbose=False,
                                                 save_to_file=False)
        cm_df = get_confusion_matrix(use_this_function=True,  # True | False
                                     classification_method=classification_method,
                                     sampling_method=sampling_method,
                                     selected_cohort=cluster,
                                     cohort_title=f'cluster_{i}',
                                     use_case_name=use_case_name,
                                     features_df=features_df,
                                     selected_features=selected_features,
                                     selected_dependent_variable=dependent_variable,
                                     use_grid_search=use_grid_search,
                                     verbose=False,
                                     save_to_file=False
                                     )
        current_settings = pd.DataFrame([{'dependent_variable': dependent_variable,
                                          'classification_method': classification_method,
                                          'cluster': f'cluster_{i}',
                                          'auc_score': auc_score,
                                          'auc_prc_score': auc_prc_score,
                                          'accuracy': get_accuracy(cm_df),
                                          'recall': get_recall(cm_df),
                                          'precision': get_precision(cm_df)
                                          }])
        classification_clusters_overview = pd.concat([classification_clusters_overview, current_settings],
                                                     ignore_index=True)

    # cleanup + transpose
    clusters_list = classification_clusters_overview.loc[:, 'cluster'].values.tolist()
    classification_clusters_overview = classification_clusters_overview.transpose()
    classification_clusters_overview.columns = clusters_list
    classification_clusters_overview.index.name = 'metrics'
    classification_clusters_overview.drop(labels=['dependent_variable', 'classification_method', 'cluster'], inplace=True)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/subgroup_analysis/clusters_overview_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            classification_clusters_overview.to_csv(output_file, index=False)
            print(f'\n STATUS: classification_clusters_overview was saved to {filename_string}')
    else:
        print('CHECK: classification_clusters_overview:')
        # print(classification_clusters_overview)

    return classification_clusters_overview