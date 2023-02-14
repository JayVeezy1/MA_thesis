import datetime
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from numpy import sort
from pandas.core.interchange import dataframe
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from step_3_data_analysis import data_visualization


def get_ids_for_cluster(avg_patient_cohort, cohort_title, features_df, selected_features,
                        selected_dependent_variable, selected_k_means_count, selected_cluster) -> list | None:
    if selected_cluster > selected_k_means_count - 1:
        print('ERROR: selected_cluster number must be < selected_k_means_count.')
        return None

    # transform df to np
    avg_np = preprocess_for_clustering(avg_patient_cohort, features_df, selected_features, selected_dependent_variable)

    # todo add: if selected_method = 'k-means' else 'dbscan'
    # get the cluster for selected_k_means_count
    k_means_list, sh_score = calculate_cluster_kmeans(avg_np, cohort_title, n_clusters=selected_k_means_count,
                                                      verbose=False)

    # connect k-means clusters back to icustay_ids
    clusters_df: dataframe = pd.DataFrame({'icustay_id': avg_patient_cohort['icustay_id'], 'cluster': k_means_list})

    print(f'CHECK: Count of patients for cluster {selected_cluster}: {len(clusters_df["icustay_id"][clusters_df["cluster"] == selected_cluster])}')

    return clusters_df['icustay_id'][clusters_df['cluster'] == selected_cluster].to_list()


def calculate_cluster_kmeans(avg_np: np.ndarray, cohort_title: str, n_clusters: int, verbose: bool = False):
    # k-means clustering: choose amount n_clusters to calculate k centroids for these clusters

    if verbose:
        print(f'STATUS: Calculating k-means on {cohort_title} for {n_clusters} clusters.')

    # Calculate KMeans
    kmeans_obj = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_  # todo: directly merge these labels back to the icustay_id -> one place. Not return of labels as list but series icustay_id | cluster_label

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
    else:
        sh_score = round(silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)

    return clustering_labels_list, sh_score


def calculate_cluster_DBSCAN(avg_np: np.ndarray, cohort_title: str, eps: float, min_samples: int,
                             verbose: bool = False):
    # density Based Spatial Clustering of Applications with Noise. Instances in dense region get clustered.

    if verbose:
        print(f'STATUS: Calculating DBSCAN on {cohort_title} for {eps} epsilon with {min_samples} min_samples.')

    # Calculate DBSCAN
    clustering_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(
        avg_np)  # we could use weights per label to give imputed labels less weight?
    clustering_labels_list = clustering_obj.labels_

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
    else:
        sh_score = round(silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)

    return clustering_labels_list, sh_score


def plot_clusters_on_3D_pacmap(plot_title, use_case_name, pacmap_data_points, cluster_count, sh_score, coloring,
                               save_to_file):
    color_map = cm.get_cmap('brg', cluster_count)  # old: tab20c

    fig = plt.figure()
    fig.tight_layout(h_pad=2, w_pad=2)
    plt.suptitle(f'{plot_title} for clusters: {cluster_count}, sh_score: {sh_score}')
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
        plt.savefig(
            f'./output/{use_case_name}/clustering/3D_clusters_kmeans_{plot_title.replace(" ", "_")}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()
    plt.close()


def preprocess_for_clustering(avg_cohort, features_df, selected_features, selected_dependent_variable):
    # Preprocessing for Clustering: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)           # keeping selected_dependent_variable for clustering?
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    avg_cohort_without_nan = avg_cohort[selected_features].fillna(0)

    return avg_cohort_without_nan.to_numpy()


def plot_sh_score_kmeans(avg_patient_cohort, cohort_title, use_case_name, features_df, selected_features,
                         selected_dependent_variable, save_to_file: bool = False):
    # This function displays the Silhouette Score curve. With this an optimal cluster count for k-means can be selected.
    print("STATUS: Calculating Silhouette Scores for k-means.")
    # Get cleaned avg_np
    avg_np = preprocess_for_clustering(avg_patient_cohort, features_df, selected_features, selected_dependent_variable)

    # Find best k-means cluster option depending on sh_score
    krange = list(range(2, 13))  # choose multiple k-means cluster options to test
    avg_silhouettes = []
    for n in krange:
        k_means_list, sh_score = calculate_cluster_kmeans(avg_np, cohort_title, n_clusters=n,
                                                          verbose=False)  # here clusters are calculated
        avg_silhouettes.append(sh_score)

    # Plot Silhouette Scores
    plt.figure(dpi=100)
    plt.title(f'Silhouette Score for k-Means on {cohort_title}')
    plt.plot(krange, avg_silhouettes)  # for DBSCAN use eps_range instead of krange
    plt.xlabel("$k$")
    plt.ylabel("Average Silhouettes Score")
    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/clustering/{f"Silhouette Score for k-Means on {cohort_title}".replace(" ", "_")}.png',
            bbox_inches="tight")
    plt.show()
    plt.close()

    return None


def plot_k_means_on_pacmap(avg_patient_cohort, cohort_title, use_case_name, features_df, selected_features,
                           selected_dependent_variable,
                           selected_cluster_count: int, save_to_file: bool = False):
    # Clean up df & transform to numpy
    avg_np = preprocess_for_clustering(avg_patient_cohort, features_df, selected_features, selected_dependent_variable)

    # PacMap needed for visualization
    pacmap_data_points, death_list = data_visualization.calculate_pacmap(avg_cohort=avg_patient_cohort,
                                                                         cohort_title=cohort_title,
                                                                         features_df=features_df,
                                                                         selected_features=selected_features,
                                                                         selected_dependent_variable=selected_dependent_variable)

    # Plot the cluster with best sh_score
    k_means_list, sh_score = calculate_cluster_kmeans(avg_np, cohort_title, n_clusters=selected_cluster_count, verbose=True)
    plot_title = f"k_Means_clusters_{selected_cluster_count} for {cohort_title}"
    plot_clusters_on_3D_pacmap(plot_title=plot_title, use_case_name=use_case_name,
                               pacmap_data_points=pacmap_data_points, cluster_count=selected_cluster_count,
                               sh_score=sh_score, coloring=k_means_list, save_to_file=save_to_file)

    return None


# todo: rework this old graphic, might be helpful for choosing optimal cluster? -> idea: quick overview of cluster details
def plot_cluster_details(plot_title: str, data: np.ndarray, death_list: list, sh_score: float, coloring: [float],
                         color_map: str, selected_dependent_variable: str, cohort_title: str, save_to_file: bool):
    fig = plt.figure()
    plt.suptitle(plot_title)
    axs = fig.subplots(2, 2)
    fig.tight_layout(h_pad=2, w_pad=2)

    # 1) Clustering
    axs[0, 0].set_title("Visualization on Pacmap")
    axs[0, 0].scatter(data[:, 0], data[:, 1], cmap=color_map, c=coloring, s=0.6, label="Patient")
    sm = matplotlib.cm.ScalarMappable(cmap=color_map,
                                      norm=matplotlib.colors.Normalize(vmin=min(coloring), vmax=max(coloring)))
    cb = fig.colorbar(sm, ax=axs[0, 0])
    # cb.set_label("Clusters")
    cb.set_ticks(list(set(coloring)))

    # 2) Silhouette Score
    axs[1, 0].set_title("Silhouette Score")
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].barh("Score", sh_score, color=plt.get_cmap("RdYlGn")(sh_score / 1))

    # 3) abs deaths per cluster
    cluster_deaths_sum = {}
    cluster_sum = {}
    for i in range(len(death_list)):
        cluster = coloring[i]
        if death_list[i]:
            if cluster not in cluster_deaths_sum.keys():
                cluster_deaths_sum[cluster] = 0
            cluster_deaths_sum[cluster] += 1

        if cluster not in cluster_sum.keys():  # total sum of cases in a cluster, might also be useful: Counter(coloring)
            cluster_sum[cluster] = 0
        cluster_sum[cluster] += 1

    axs[1, 1].set_title(f"total {selected_dependent_variable} per cluster")
    axs[1, 1].set_xticks(list(set(coloring)))
    for cluster in cluster_deaths_sum.keys():
        axs[1, 1].bar(cluster, cluster_deaths_sum[cluster],
                      color=plt.get_cmap(color_map)(
                          cluster / max(coloring)))  # can raise a warning if "highest" cluster is 0

    # 4) rel deaths per cluster
    axs[0, 1].set_title(f"relative {selected_dependent_variable} per cluster")
    axs[0, 1].set_xticks(list(set(coloring)))
    axs[0, 1].set_ylim(0, 1.0)
    for cluster in cluster_deaths_sum.keys():
        axs[0, 1].bar(cluster, cluster_deaths_sum[cluster] / cluster_sum[cluster],
                      color=plt.get_cmap(color_map)(cluster / max(coloring)))

    if save_to_file:
        plt.savefig(f'./output/clustering/overview_kmeans_{cohort_title}.png')
    plt.show()
    plt.close()

    return None


def get_features_overview(selected_cohort, selected_features, features_df, cohort_title):
    # creates the base for the base features_overview_table -> Variables | Classification(bins) | Count(complete_set)
    factorization_df = pd.read_csv(f'./supplements/factorization_table_{cohort_title}.csv')

    current_overview_table: dataframe = pd.DataFrame({'Variables': 'total_count',
                                                      'Classification': 'icustay_ids',
                                                      'complete_set': [selected_cohort['icustay_id'].count()],
                                                      })

    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    features_to_factorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_factorize = [x for x in features_to_factorize if
                             x not in features_to_remove]  # drop features_to_remove from factorization

    # This is used only for the first creation of the clusters_overview_table for the 'complete_set' case
    for feature in selected_features:
        # normal case, no binning needed
        if features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'False':
            # use unfactorized name from supplements factorization_table
            if feature in features_to_factorize:
                for appearance in sort(pd.unique(selected_cohort[feature])):
                    try:
                        appearance_name = factorization_df.loc[(factorization_df['feature'] == feature) & (
                                factorization_df['factorized_values'] == appearance), 'unfactorized_value'].item()
                    except ValueError as e:
                        appearance_name = 'no_data'
                    temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [appearance_name],
                                                       'complete_set': [selected_cohort[feature][
                                                                           selected_cohort[
                                                                               feature] == appearance].count()],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)
            else:
                for appearance in sort(pd.unique(selected_cohort[feature])):
                    temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [appearance],
                                                       'complete_set': [selected_cohort[feature][
                                                                           selected_cohort[
                                                                               feature] == appearance].count()],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)
        # binning needed for vital signs, etc.
        elif features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'True':
            try:
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                feature_min = int(np.nanmin(selected_cohort[feature].values))
                feature_max = int(np.nanmax(selected_cohort[feature].values))

                if feature_min == feature_max:
                    feature_appearances_series = selected_cohort[feature].value_counts(bins=[feature_min,
                                                                                             feature_max])
                else:
                    feature_appearances_series = selected_cohort[feature].value_counts(bins=[feature_min,
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
                    temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                       'Classification': [str(binning_intervals[i])],
                                                       'complete_set': [binning_counts[i]],
                                                       })
                    current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)

            except ValueError as e:  # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column {feature} probably is all-NaN or only one entry. Error-Message: {e}')
                temp_df: dataframe = pd.DataFrame({'Variables': [feature],
                                                   'Classification': ['All Entries NaN'],
                                                   'complete_set': [0],
                                                   })
                current_overview_table = pd.concat([current_overview_table, temp_df], ignore_index=True)

    return current_overview_table


def get_overview_for_cluster(cluster_cohort, selected_features, features_df, current_overview_table: dataframe,
                             complete_cohort,
                             selected_cluster, cohort_title):
    # Adds new columns to the features_overview_table for each cluster

    # total_count row
    current_overview_table.loc[(current_overview_table['Variables'] == 'total_count') & (current_overview_table['Classification'] == 'icustay_ids'), f'cluster_{selected_cluster}'] = cluster_cohort['icustay_id'].count()

    # get features_to_factorize
    factorization_df = pd.read_csv(f'./supplements/factorization_table_{cohort_title}.csv')
    features_to_remove = features_df['feature_name'].loc[features_df['must_be_removed'] == 'yes'].to_list()
    features_to_factorize = features_df['feature_name'].loc[
        features_df['categorical_or_continuous'] == 'categorical'].to_list()
    features_to_factorize = [x for x in features_to_factorize if
                             x not in features_to_remove]  # drop features_to_remove from factorization

    # current_overview_table[f'cluster_{selected_cluster}'] = np.nan
    for feature in selected_features:
        # normal case, no binning needed
        if features_df['needs_binning'][features_df[
                                            'feature_name'] == feature].item() == 'False':  # todo: use avg_preprocessed_cohort also in other elif cases, but 'feature_name' needed inside avg_preprocessed_cohort

            # use unfactorized name from supplements factorization_table
            if feature in features_to_factorize:
                for appearance in sort(pd.unique(cluster_cohort[feature])):
                    try:
                        appearance_name = factorization_df.loc[(factorization_df['feature'] == feature) & (
                                factorization_df['factorized_values'] == appearance), 'unfactorized_value'].item()
                    except ValueError as e:
                        appearance_name = 'no_data'
                    current_overview_table.loc[(current_overview_table['Variables'] == feature) & (
                            current_overview_table[
                                'Classification'] == appearance_name), f'cluster_{selected_cluster}'] = [
                        (cluster_cohort[feature][cluster_cohort[feature] == appearance].count())]
            else:
                for appearance in sort(pd.unique(cluster_cohort[feature])):
                    current_overview_table.loc[(current_overview_table['Variables'] == feature) & (
                            current_overview_table[
                                'Classification'] == appearance), f'cluster_{selected_cluster}'] = [
                        (cluster_cohort[feature][cluster_cohort[feature] == appearance].count())]

        # binning needed for vital signs, etc.
        elif features_df['needs_binning'][features_df['feature_name'] == feature].item() == 'True':
            try:
                warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                feature_min = int(np.nanmin(
                    complete_cohort[feature].values))  # important: use complete_cohort_preprocessed here!
                feature_max = int(np.nanmax(complete_cohort[feature].values))

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
                        binning_intervals[i])), f'cluster_{selected_cluster}'] = binning_counts[i]

            except ValueError as e:  # this happens if for the selected cohort (a small cluster) all patients have NaN
                print(f'WARNING: Column {feature} probably is all-NaN or only one entry. Error-Message: {e}')
                current_overview_table.loc[
                    (current_overview_table['Variables'] == feature), f'cluster_{selected_cluster}'] = 0

    # Cleanup of NaN
    current_overview_table.loc[current_overview_table[f'cluster_{selected_cluster}'].isnull(), f'cluster_{selected_cluster}'] = 0

    return current_overview_table


def calculate_clusters_overview_table(selected_cohort, cohort_title, use_case_name,
                                      selected_clusters_count, features_df,
                                      selected_features, selected_dependent_variable, save_to_file: False):
    # step 1: get counts for complete dataset -> based on general_statistics.calculate_feature_overview_table
    features_overview_table = get_features_overview(selected_cohort=selected_cohort,
                                                    selected_features=selected_features,
                                                    features_df=features_df,
                                                    cohort_title=cohort_title)

    for selected_cluster in range(0, selected_clusters_count):
        # step 2: get each cluster as df
        filtered_cluster_icustay_ids: list = get_ids_for_cluster(
            avg_patient_cohort=selected_cohort,
            cohort_title='selected_patient_cohort',
            features_df=features_df,
            selected_features=selected_features,
            selected_dependent_variable=selected_dependent_variable,
            selected_k_means_count=selected_clusters_count,
            selected_cluster=selected_cluster)
        filtered_cluster_cohort = selected_cohort[
            selected_cohort['icustay_id'].isin(filtered_cluster_icustay_ids)]

        # step 3: get count of occurrences per bin for this cluster
        # todo: add row 'Total-Count'
        features_overview_table: dataframe = get_overview_for_cluster(cluster_cohort=filtered_cluster_cohort,
                                                                      complete_cohort=selected_cohort,
                                                                      selected_features=selected_features,
                                                                      features_df=features_df,
                                                                      current_overview_table=features_overview_table,
                                                                      selected_cluster=selected_cluster,
                                                                      cohort_title=cohort_title)

    if save_to_file:
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        filename_string: str = f'./output/{use_case_name}/clustering/clusters_overview_table_{cohort_title}_{current_time}.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            features_overview_table.to_csv(output_file, index=False)
            print(f'STATUS: clusters_overview_table was saved to {filename_string}')
    else:
        print('CHECK: features_overview_table:')
        print(features_overview_table)

    return None
