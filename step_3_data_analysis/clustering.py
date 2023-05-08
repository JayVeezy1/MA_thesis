import datetime
import math
import warnings

import streamlit as st
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from numpy import sort
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes

from step_2_preprocessing.preprocessing_functions import get_one_hot_encoding
from step_3_data_analysis import data_visualization

@st.cache_data
def preprocess_for_clustering(selected_cohort, features_df, selected_features, selected_dependent_variable,
                              use_encoding: False):
    # Removal of known features_to_remove
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    selected_features = [x for x in selected_features if x not in features_to_remove]

    # Preprocessing for Clustering: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)
    selected_cohort = selected_cohort[selected_features].fillna(0)

    if use_encoding:
        # Encoding of categorical features (one hot encoding)
        categorical_features = features_df['feature_name'].loc[
            features_df['categorical_or_continuous'] == 'categorical'].to_list()
        categorical_features = [x for x in categorical_features if x in selected_features]
        selected_cohort = get_one_hot_encoding(selected_cohort, categorical_features)

    try:
        selected_cohort.drop(columns='icustay_id', inplace=True)
    except KeyError as e:
        pass

    # print(f'CHECK: {len(selected_features)} features used for Clustering.')

    return selected_cohort  # TODO: this was removed: turn back? .to_numpy()        # dependent_variable and icustay_id still inside, will be used and removed outside


def get_ids_for_cluster(avg_patient_cohort, cohort_title, features_df, selected_features, selected_dependent_variable,
                        selected_k_means_count, selected_cluster, use_encoding, verbose: False) -> list | None:
    if selected_cluster > selected_k_means_count - 1:
        print('ERROR: selected_cluster number must be < selected_k_means_count.')
        return None

    # transform df to np
    selected_cohort = preprocess_for_clustering(selected_cohort=avg_patient_cohort,
                                                features_df=features_df,
                                                selected_features=selected_features.copy(),
                                                selected_dependent_variable=selected_dependent_variable,
                                                use_encoding=use_encoding)  # use selected_features.copy() because in function other dependent variables are remove

    # get the cluster for selected_k_means_count -> currently this function is not for DBSCAN, because dynamic cluster count there
    k_means_list, sh_score, inertia = calculate_cluster_kmeans(selected_cohort, cohort_title,
                                                               n_clusters=selected_k_means_count,
                                                               verbose=False)

    # connect k-means clusters back to icustay_ids
    clusters_df = pd.DataFrame({'icustay_id': avg_patient_cohort['icustay_id'], 'cluster': k_means_list})

    if verbose:
        print(
            f'CHECK: Count of patients for cluster {selected_cluster}: {len(clusters_df["icustay_id"][clusters_df["cluster"] == selected_cluster])}')

    return clusters_df['icustay_id'][clusters_df['cluster'] == selected_cluster].to_list()


def plot_clusters_on_3D_pacmap(plot_title, use_case_name, pacmap_data_points, cluster_count, sh_score, coloring,
                               save_to_file):
    if cluster_count > 15:
        cluster_count = 15
    color_map = cm.get_cmap('brg', cluster_count)  # old: tab20c

    fig = plt.figure()
    fig.tight_layout(h_pad=2, w_pad=2)
    plt.suptitle(f'{plot_title} for clusters: {cluster_count}, sh_score: {sh_score}', wrap=True)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    ax1.scatter(pacmap_data_points[:, 0], pacmap_data_points[:, 1], pacmap_data_points[:, 2], cmap=color_map,
                c=coloring, s=0.7, label='Patient')

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=color_map, norm=matplotlib.colors.Normalize(vmin=min(coloring),
                                                                                                    vmax=max(
                                                                                                        coloring))),
                      ax=ax1)
    cb.set_label('Clusters')
    cb.set_ticks(list(set(coloring)))
    plt.legend()

    if save_to_file:
        plt.show()
        plt.savefig(
            f'./output/{use_case_name}/clustering/clustering_{plot_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png',
            dpi=600)
    # plt.close()

    return plt


@st.cache_data
def calculate_cluster_kmeans(selected_cohort, cohort_title: str, n_clusters: int, verbose: bool = False):
    # k-means clustering: choose amount n_clusters to calculate k centroids for these clusters
    if verbose:
        print(f'STATUS: Calculating k-means on {cohort_title} for {n_clusters} clusters.')

    avg_np = selected_cohort.to_numpy()

    # Calculate KMeans
    kmeans_obj = KMeans(init='k-means++', n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
        inertia = (-1)
    else:
        sh_score = round(silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)
        inertia = kmeans_obj.inertia_

    return clustering_labels_list, sh_score, inertia


@st.cache_data
def calculate_cluster_kprot(selected_cohort, cohort_title: str, selected_features, n_clusters: int,
                            verbose: bool = False):
    # k-prototypes clustering: calculate clusters also for categorical values
    if verbose:
        print(f'STATUS: Calculating k-prototypes on {cohort_title} for {n_clusters} clusters.')

    # Transform df to numpy array-like, needed for clustering
    avg_np = selected_cohort.to_numpy()

    # Get categorical_index
    categorical_features = ['admission_type', 'dbsource', 'ethnicity', 'gender', 'insurance', 'marital_status',
                            'religion', 'stroke_type', 'cancer_flag', 'sepsis_flag', 'obesity_flag', 'hypertension_flag',
                            'diabetes_flag', 'electivesurgery', 'mechvent', 'language', 'discharge_location']
    categorical_features = [x for x in categorical_features if x in selected_features]
    categorical_index: list = []
    available_columns: list = selected_cohort.columns.to_list()
    for feature in categorical_features:
        try:
            categorical_index.append(available_columns.index(feature))
        except ValueError:  # occurs when available_columns does not contain one of the categorical features
            pass

    # Calculate KProto
    kprot_obj = KPrototypes(n_clusters=n_clusters, init='Cao', random_state=0, max_iter=350)
    kprot_clustering = kprot_obj.fit(X=avg_np, categorical=categorical_index)  # maybe use n_jobs=4 for parallelization
    clustering_labels_list = kprot_clustering.labels_

    # get sh_score
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
        inertia = (-1)
    else:
        sh_score = round(silhouette_score(X=avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0),
                         2)
        inertia = kprot_obj.cost_
        # todo future research: only cost_ available instead of inertia_ for KPROT
        # but cost = sum of distances to cluster and inertia = squared-sum of distances to cluster
        # might be good to calculate inertia, but ^2 is monotonous, transformation not really needed to assess clusters

    return clustering_labels_list, sh_score, inertia


def plot_sh_score(use_this_function: False, selected_cohort, cohort_title, use_case_name, features_df,
                  selected_features, selected_dependent_variable, use_encoding: False, clustering_method,
                  save_to_file: bool = False):
    if not use_this_function:
        return None

    # This function displays the Silhouette Score curve. With this an optimal cluster count for k-means can be selected.
    print(f'STATUS: Calculating Silhouette Scores for {clustering_method}.')
    # Get cleaned avg_np
    selected_cohort = preprocess_for_clustering(selected_cohort=selected_cohort,
                                                features_df=features_df,
                                                selected_features=selected_features,
                                                use_encoding=use_encoding,
                                                selected_dependent_variable=selected_dependent_variable)

    # Find best k-means cluster option depending on sh_score -> check plot manually
    krange = list(range(2, 15))  # choose multiple k-means cluster options to test
    avg_silhouettes = []
    inertias = []

    # probably cleaner to have if outside for-loop
    for n in krange:
        if clustering_method == 'kmeans':
            k_means_list, sh_score, inertia = calculate_cluster_kmeans(selected_cohort, cohort_title, n_clusters=n,
                                                                       verbose=False)  # here clusters are calculated
        elif clustering_method == 'kprot':
            k_means_list, sh_score, inertia = calculate_cluster_kprot(selected_cohort=selected_cohort,
                                                                      selected_features=selected_features,
                                                                      cohort_title=cohort_title, n_clusters=n,
                                                                      verbose=False)  # here clusters are calculated
        else:
            print(f'WARNING: Unknown clustering method: {clustering_method}. sh_score was not calculated.')
            return None
        avg_silhouettes.append(sh_score)  # silhouette score should be maximal
        inertias.append(inertia)  # inertia = distortion = Sum-of-Squared-Errors = Elbow method, should be minimal

    # Silhouette Score + Plot Elbow/Inertia Method
    fig, ax = plt.subplots()
    inertia_color = '#0000B0'
    ax.plot(krange, inertias, color=inertia_color, marker=".")
    ax.set_xlabel('k', fontsize=14)
    ax.set_ylabel('Distortion (SSE)', color=inertia_color, fontsize=14)
    ax2 = ax.twinx()  # twin object for second y-axis
    sh_color = '#B00000'
    ax2.plot(krange, avg_silhouettes, color=sh_color, marker=".")
    ax2.set_ylabel('Silhouette Score', color=sh_color, fontsize=14)
    plt.title(f'SSE and Silhouette Score for {clustering_method} on {cohort_title}', wrap=True)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        plt.savefig(f'./output/{use_case_name}/clustering/optimal_{clustering_method}_cluster_count_{cohort_title}_{current_time}.png',
                    bbox_inches='tight',
                    dpi=600)
        plt.show()
    # plt.close()

    return plt


def plot_k_means_on_pacmap(use_this_function: False, display_sh_score: False, selected_cohort, cohort_title,
                           use_case_name, features_df,
                           selected_features, selected_dependent_variable, selected_cluster_count, use_encoding: False,
                           save_to_file: False):
    if not use_this_function:
        return None

    if display_sh_score:
        # check optimal cluster count with sh_score and distortion (SSE) - can be bad if too many features selected
        plot_sh_score(use_this_function=True,  # True | False
                      selected_cohort=selected_cohort,
                      cohort_title=cohort_title,
                      use_case_name=use_case_name,
                      features_df=features_df,
                      selected_features=selected_features,
                      selected_dependent_variable=selected_dependent_variable,
                      use_encoding=use_encoding,
                      clustering_method='kmeans',
                      save_to_file=save_to_file)


    # PacMap needed for visualization
    pacmap_data_points, death_list = data_visualization.calculate_pacmap(selected_cohort=selected_cohort,
                                                                         cohort_title=cohort_title,
                                                                         features_df=features_df,
                                                                         selected_features=selected_features,
                                                                         selected_dependent_variable=selected_dependent_variable,
                                                                         use_encoding=use_encoding)

    # Clean up df & transform to numpy
    selected_cohort_preprocessed = preprocess_for_clustering(selected_cohort, features_df, selected_features,
                                                selected_dependent_variable,
                                                use_encoding)

    # Plot the cluster with best sh_score
    k_means_list, sh_score, inertia = calculate_cluster_kmeans(selected_cohort=selected_cohort_preprocessed,
                                                               cohort_title=cohort_title,
                                                               n_clusters=selected_cluster_count,
                                                               verbose=True)
    plot_title = f'k_Means_{cohort_title}'
    kmeans_plot = plot_clusters_on_3D_pacmap(plot_title=plot_title, use_case_name=use_case_name,
                               pacmap_data_points=pacmap_data_points, cluster_count=selected_cluster_count,
                               sh_score=sh_score, coloring=k_means_list, save_to_file=save_to_file)

    return kmeans_plot


def get_clusters_overview_table(original_cohort, selected_features, features_df, cohort_title):
    # creates the base for the base features_overview_table -> Variables | Classification(bins) | Count(complete_set)

    current_overview_table = pd.DataFrame({'Variables': 'total_count',
                                                      'Classification': 'icustay_ids',
                                                      'complete_set': [original_cohort['icustay_id'].count()],
                                                      })

    # get features_to_factorize
    factorization_df = pd.read_excel(f'./supplements/FACTORIZATION_TABLE.xlsx')
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    features_to_refactorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_refactorize = [x for x in features_to_refactorize if
                               x not in features_to_remove]  # drop features_to_remove from factorization

    # This is used only for the first creation of the clusters_overview_table for the 'complete_set' case
    for feature in selected_features:
        # normal case, no binning needed
        if features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'False':
            # use unfactorized name from supplements factorization_table
            if feature in features_to_refactorize:
                for appearance in sort(pd.unique(original_cohort[feature])):

                    if math.isnan(appearance):
                        break
                    temp_fact_df = factorization_df.loc[factorization_df['feature'] == feature]
                    temp_index = temp_fact_df['factorized_value'] == appearance
                    try:
                        appearance_name = temp_fact_df.loc[temp_index, 'unfactorized_value'].item()
                    except ValueError as e:
                        # print(f'CHECK: multiple unfactorized_values for feature {feature}.')
                        appearance_name = temp_fact_df.loc[
                            temp_index, 'unfactorized_value']  # simply use first available unfactorized_value
                        appearance_name = appearance_name.iloc[0] + '_GROUP'

                    temp_df = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [appearance_name],
                                                       'complete_set': [original_cohort[feature][
                                                                            original_cohort[
                                                                                feature] == appearance].count()],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)
            else:
                for appearance in sort(pd.unique(original_cohort[feature])):
                    temp_df = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [appearance],
                                                       'complete_set': [original_cohort[feature][
                                                                            original_cohort[
                                                                                feature] == appearance].count()],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)
        # binning needed for vital signs, etc.
        elif features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'True':
            try:
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                feature_min = int(np.nanmin(original_cohort[feature].values))
                feature_max = int(np.nanmax(original_cohort[feature].values))

                if feature_min == feature_max:
                    feature_appearances_series = original_cohort[feature].value_counts(bins=[feature_min,
                                                                                             feature_max])
                else:
                    feature_appearances_series = original_cohort[feature].value_counts(bins=[feature_min,
                                                                                             feature_min + round(
                                                                                                 (
                                                                                                         feature_max - feature_min) * 1 / 3,
                                                                                                 2),
                                                                                             feature_min + round(
                                                                                                 (
                                                                                                         feature_max - feature_min) * 2 / 3,
                                                                                                 2),
                                                                                             feature_max])
                feature_appearances_df = pd.DataFrame()
                feature_appearances_df['intervals'] = feature_appearances_series.keys()
                feature_appearances_df['counts'] = feature_appearances_series.values
                feature_appearances_df['interval_starts'] = feature_appearances_df['intervals'].map(lambda x: x.left)
                feature_appearances_df = feature_appearances_df.sort_values(by='interval_starts')
                binning_intervals: list = feature_appearances_df['intervals'].to_list()
                binning_counts: list = feature_appearances_df['counts'].to_list()

                for i in range(0, len(binning_intervals)):
                    temp_df = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [str(binning_intervals[i])],
                                                       'complete_set': [binning_counts[i]],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)

            except ValueError as e:  # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column {feature} probably is all-NaN or only one entry. Error-Message: {e}')
                temp_df = pd.DataFrame({'Variables': [feature],
                                                   'Classification': ['All Entries NaN'],
                                                   'complete_set': [0],
                                                   })
                current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)

    return current_overview_table


def get_overview_for_cluster(cluster_cohort, selected_features, features_df, current_overview_table,
                             original_cohort, selected_cluster_number, cohort_title):
    # Adds new columns to the features_overview_table for each cluster

    # total_count row
    current_overview_table.loc[(current_overview_table['Variables'] == 'total_count') & (
            current_overview_table['Classification'] == 'icustay_ids'), f'cluster_{selected_cluster_number}'] = \
        cluster_cohort['icustay_id'].count()

    # get features_to_factorize
    factorization_df = pd.read_excel(f'./supplements/FACTORIZATION_TABLE.xlsx')
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    features_to_refactorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_refactorize = [x for x in features_to_refactorize if
                               x not in features_to_remove]  # drop features_to_remove from factorization

    for feature in selected_features:
        # normal case, no binning needed
        if features_df['needs_binning'][features_df[
                                            'feature_name'] == feature].item() == 'False':

            # use unfactorized name from supplements factorization_table
            if feature in features_to_refactorize:
                for appearance in sort(pd.unique(cluster_cohort[feature])):

                    if math.isnan(appearance):
                        break
                    temp_fact_df = factorization_df.loc[factorization_df['feature'] == feature]
                    temp_index = temp_fact_df['factorized_value'] == appearance
                    try:
                        appearance_name = temp_fact_df.loc[temp_index, 'unfactorized_value'].item()
                    except ValueError as e:
                        # print(f'CHECK: multiple unfactorized_values for feature {feature}.')
                        appearance_name = temp_fact_df.loc[temp_index, 'unfactorized_value']    # simply use first available unfactorized_value
                        appearance_name = appearance_name.iloc[0] + '_GROUP'

                    current_overview_table.loc[(current_overview_table['Variables'] == feature) & (
                            current_overview_table[
                                'Classification'] == appearance_name), f'cluster_{selected_cluster_number}'] = [
                        (cluster_cohort[feature][cluster_cohort[feature] == appearance].count())]
            else:
                for appearance in sort(pd.unique(cluster_cohort[feature])):
                    current_overview_table.loc[(current_overview_table['Variables'] == feature) & (
                            current_overview_table[
                                'Classification'] == appearance), f'cluster_{selected_cluster_number}'] = [
                        (cluster_cohort[feature][cluster_cohort[feature] == appearance].count())]

        # binning needed for vital signs, etc.
        elif features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'True':
            try:
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                feature_min = int(np.nanmin(
                    original_cohort[feature].values))  # important: use complete_cohort_preprocessed here!
                feature_max = int(np.nanmax(original_cohort[feature].values))

                if feature_min == feature_max:
                    feature_appearances_series = cluster_cohort[feature].value_counts(bins=[feature_min,
                                                                                            feature_max])
                else:
                    feature_appearances_series = cluster_cohort[feature].value_counts(bins=[feature_min,
                                                                                            feature_min + round(
                                                                                                (
                                                                                                        feature_max - feature_min) * 1 / 3,
                                                                                                2),
                                                                                            feature_min + round(
                                                                                                (
                                                                                                        feature_max - feature_min) * 2 / 3,
                                                                                                2),
                                                                                            feature_max])
                feature_appearances_df = pd.DataFrame()
                feature_appearances_df['intervals'] = feature_appearances_series.keys()
                feature_appearances_df['counts'] = feature_appearances_series.values
                feature_appearances_df['interval_starts'] = feature_appearances_df['intervals'].map(lambda x: x.left)
                feature_appearances_df = feature_appearances_df.sort_values(by='interval_starts')
                binning_intervals: list = feature_appearances_df['intervals'].to_list()
                binning_counts: list = feature_appearances_df['counts'].to_list()

                for i in range(0, len(binning_intervals)):
                    current_overview_table.loc[(current_overview_table['Variables'] == feature) & (
                            current_overview_table['Classification'] == str(
                        binning_intervals[i])), f'cluster_{selected_cluster_number}'] = binning_counts[i]

            except ValueError as e:  # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column {feature} probably is all-NaN or only one entry. Error-Message: {e}')
                current_overview_table.loc[
                    (current_overview_table['Variables'] == feature), f'cluster_{selected_cluster_number}'] = 0

    # Cleanup of NaN
    current_overview_table.loc[
        current_overview_table[f'cluster_{selected_cluster_number}'].isnull(), f'cluster_{selected_cluster_number}'] = 0

    return current_overview_table


def get_kmeans_clusters(original_cohort, features_df, selected_features, selected_dependent_variable,
                        selected_k_means_count, use_encoding, verbose: False):
    # returns a list of clusters [dataframe, ...]
    kmeans_clusters: list = []
    for selected_cluster in range(0, selected_k_means_count):
        filtered_cluster_icustay_ids: list = get_ids_for_cluster(
            avg_patient_cohort=original_cohort,
            cohort_title='clustered_cohort',
            features_df=features_df,
            selected_features=selected_features,
            selected_dependent_variable=selected_dependent_variable,
            selected_k_means_count=selected_k_means_count,
            selected_cluster=selected_cluster,
            use_encoding=use_encoding,
            verbose=verbose)

        filtered_cluster_cohort = original_cohort[original_cohort['icustay_id'].isin(filtered_cluster_icustay_ids)]
        kmeans_clusters.append(filtered_cluster_cohort)

    return kmeans_clusters


def calculate_clusters_overview_table(use_this_function: False, selected_cohort, cohort_title, use_case_name,
                                      features_df, selected_features, selected_dependent_variable,
                                      selected_k_means_count, use_encoding: False, save_to_file: False):
    # currently this function is only usable for manually selected cluster_count -> kmeans but not DBSCAN
    if not use_this_function:
        return None

    # step 1: get counts for complete dataset -> based on general_statistics.calculate_feature_overview_table
    clusters_overview_table = get_clusters_overview_table(original_cohort=selected_cohort,
                                                          selected_features=selected_features,
                                                          features_df=features_df,
                                                          cohort_title=cohort_title)

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
                                                                      cohort_title=cohort_title)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/clustering/clusters_overview_table_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            clusters_overview_table.to_csv(output_file, index=False)
            print(f'STATUS: clusters_overview_table was saved to {filename_string}')
    else:
        print('CHECK: clusters_overview_table:')
        print(clusters_overview_table)

    return None

@st.cache_data
def calculate_cluster_dbscan(selected_cohort, eps, min_samples, cohort_title, verbose: bool = False):
    if verbose:
        print(f'STATUS: Calculating DBSCAN on {cohort_title} for {eps} epsilon with {min_samples} min_samples.')

    avg_np = selected_cohort.to_numpy()

    # Calculate DBSCAN
    clustering_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(avg_np)
    clustering_labels_list = clustering_obj.labels_

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
    else:
        sh_score = round(
            silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)

    return clustering_labels_list, sh_score


def plot_sh_score_DBSCAN(use_this_function, selected_cohort, cohort_title, use_case_name, features_df,
                         selected_features, selected_dependent_variable, selected_eps, selected_min_sample, use_encoding: False, save_to_file):
    if not use_this_function:
        return None

    # This function displays the Silhouette Score curve. With this an optimal cluster count for k-means can be selected.
    print('STATUS: Calculating Silhouette Scores for DBSCAN.')
    # Get cleaned avg_np
    selected_cohort = preprocess_for_clustering(selected_cohort, features_df, selected_features,
                                                selected_dependent_variable,
                                                use_encoding)

    # Find best sh_score DBSCAN parameters epsilon and min_samples based on predetermined lists
    if selected_eps is None or selected_min_sample is None:
        eps_range = [0.25, 0.5, 0.75, 1]  # eps = radius around a node
        min_samples = [5, 10, 20, 100]  # min_samples = number of neighbors to be considered central-node
        best_sh_score = 0
        best_eps = 0
        best_min_sample = 0
        best_avg_silhouettes = []
        for min_sample in min_samples:
            current_avg_silhouettes = []
            for eps in eps_range:
                db_scan_list, temp_sh_score = calculate_cluster_dbscan(selected_cohort, eps=eps, min_samples=min_sample,
                                                                       cohort_title=cohort_title)
                current_avg_silhouettes.append(temp_sh_score)
                # look for best setting
                if temp_sh_score > 0:
                    if temp_sh_score > best_sh_score:
                        best_sh_score = temp_sh_score
                        best_eps = eps
                        best_min_sample = min_sample
                        best_avg_silhouettes = current_avg_silhouettes
                else:
                    print(f'sh score was not above 0, no good clustering for eps: {eps}, min_samples: {min_sample}')
        # print best setting
        print(
            f'CHECK: Best sh score{best_sh_score} was reached with settings eps: {best_eps} and min_sample: {best_min_sample}')
    else:
        eps_range = [0.01, selected_eps, 2*selected_eps]
        best_sh_score = 0
        best_eps = 0
        best_min_sample = 0
        current_avg_silhouettes = []
        for eps in eps_range:
            db_scan_list, temp_sh_score = calculate_cluster_dbscan(selected_cohort, eps=eps, min_samples=selected_min_sample,
                                                                   cohort_title=cohort_title)
            current_avg_silhouettes.append(temp_sh_score)
        best_avg_silhouettes = current_avg_silhouettes
        best_min_sample = selected_min_sample

    # Plot Silhouette Scores -> not in other function, because no SSE available
    plt.figure(dpi=100)
    plt.title(f'Silhouette Scores, DBSCAN on {cohort_title}, min_sample: {best_min_sample}', wrap=True)
    plt.plot(eps_range, best_avg_silhouettes)  # for DBSCAN use eps_range instead of krange
    plt.xlabel("$eps$")
    plt.ylabel("Average Silhouettes Score")
    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/clustering/silhouette_score_DBSCAN_{cohort_title}.png',
            dpi=600,
            bbox_inches='tight')
        plt.show()

    # plt.close()

    return plt


def plot_DBSCAN_on_pacmap(use_this_function: bool, display_sh_score: False, selected_cohort,
                          cohort_title: str, use_case_name: str,
                          features_df, selected_features: list, selected_dependent_variable: str,
                          selected_eps, selected_min_sample, use_encoding: False, save_to_file: bool):
    if not use_this_function:
        return None

    if display_sh_score:
        plot_sh_score_DBSCAN(use_this_function=True,  # True | False
                             selected_cohort=selected_cohort,
                             cohort_title=cohort_title,
                             use_case_name=use_case_name,
                             features_df=features_df,
                             selected_features=selected_features,
                             selected_dependent_variable=selected_dependent_variable,
                             selected_eps=selected_eps,     # alternative: input 'None' for eps and min_sample
                             selected_min_sample=selected_min_sample,
                             use_encoding=use_encoding,
                             save_to_file=save_to_file)


    # PacMap needed for visualization
    pacmap_data_points, death_list = data_visualization.calculate_pacmap(selected_cohort=selected_cohort,
                                                                         cohort_title=cohort_title,
                                                                         features_df=features_df,
                                                                         selected_features=selected_features,
                                                                         selected_dependent_variable=selected_dependent_variable,
                                                                         use_encoding=use_encoding)
    # Clean up df & transform to numpy
    selected_cohort_preprocessed = preprocess_for_clustering(selected_cohort, features_df, selected_features,
                                                             selected_dependent_variable,
                                                             use_encoding)

    # Plot the cluster with best sh_score
    dbscan_list, sh_score = calculate_cluster_dbscan(selected_cohort_preprocessed, eps=selected_eps,
                                                     min_samples=selected_min_sample,
                                                     cohort_title=cohort_title)
    plot_title = f'DBSCAN_{cohort_title}_eps_{selected_eps}_min_sample_{selected_min_sample}'
    dbscan_plot = plot_clusters_on_3D_pacmap(plot_title=plot_title, use_case_name=use_case_name,
                               pacmap_data_points=pacmap_data_points, cluster_count=len(set(dbscan_list)),
                               sh_score=sh_score, coloring=dbscan_list, save_to_file=save_to_file)

    return dbscan_plot, dbscan_list


def plot_k_prot_on_pacmap(use_this_function, display_sh_score, selected_cohort, cohort_title, use_case_name,
                          features_df, selected_features, selected_dependent_variable, selected_cluster_count,
                          use_encoding, save_to_file):
    if not use_this_function:
        return None

    if display_sh_score:
        # check optimal cluster count with sh_score and distortion (SSE) - can be bad if too many features selected
        plot_sh_score(use_this_function=True,  # True | False
                      selected_cohort=selected_cohort,
                      cohort_title=cohort_title,
                      use_case_name=use_case_name,
                      features_df=features_df,
                      selected_features=selected_features,
                      selected_dependent_variable=selected_dependent_variable,
                      use_encoding=use_encoding,
                      clustering_method='kprot',
                      save_to_file=save_to_file)

    # PacMap needed for visualization
    pacmap_data_points, death_list = data_visualization.calculate_pacmap(selected_cohort=selected_cohort,
                                                                         cohort_title=cohort_title,
                                                                         features_df=features_df,
                                                                         selected_features=selected_features,
                                                                         selected_dependent_variable=selected_dependent_variable,
                                                                         use_encoding=use_encoding)

    # Clean up df
    selected_cohort_preprocessed = preprocess_for_clustering(selected_cohort, features_df, selected_features,
                                                             selected_dependent_variable,
                                                             use_encoding=False)

    # Plot the cluster with best sh_score
    k_prot_list, sh_score, inertia = calculate_cluster_kprot(selected_cohort=selected_cohort_preprocessed,
                                                             cohort_title=cohort_title,
                                                             selected_features=selected_features,
                                                             n_clusters=selected_cluster_count,
                                                             verbose=True)
    plot_title = f'k_prototypes_{cohort_title}_{selected_cluster_count}_clusters'
    kprot_plot = plot_clusters_on_3D_pacmap(plot_title=plot_title, use_case_name=use_case_name,
                               pacmap_data_points=pacmap_data_points, cluster_count=selected_cluster_count,
                               sh_score=sh_score, coloring=k_prot_list, save_to_file=save_to_file)

    return kprot_plot
