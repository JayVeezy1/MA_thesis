import datetime

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from pandas.core.interchange import dataframe
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from step_2_preprocessing import preprocessing_functions
from step_3_data_analysis import data_visualization


def get_ids_for_cluster(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable,
                        selected_k_means_count,
                        selected_cluster, filter_labels: bool = False) -> list | None:
    if selected_cluster > selected_k_means_count - 1:
        print('ERROR: selected_cluster number must be < selected_k_means_count.')
        return None

    avg_np = transform_df_to_np_for_clustering(avg_patient_cohort, selected_features, selected_dependent_variable,
                                               filter_labels)

    # add: if selected_method = 'k-means' else 'dbscan'
    k_means_list, sh_score = calculate_cluster_kmeans(avg_np, cohort_title, n_clusters=selected_k_means_count,
                                                      verbose=True)

    # connect k-means clusters back to icustay_ids
    # todo: recheck if this connection is correct, if sorting changed then wrong icustay_id to wrong cluster
    clusters_df: dataframe = pd.DataFrame({'icustay_id': avg_patient_cohort['icustay_id'],'cluster': k_means_list})

    print(
        f'CHECK: Count of patients for cluster {selected_cluster}: {len(clusters_df["icustay_id"][clusters_df["cluster"] == selected_cluster])}')

    return clusters_df['icustay_id'][clusters_df['cluster'] == selected_cluster].to_list()


def transform_df_to_np_for_clustering(avg_patient_cohort,
                                      selected_features,
                                      selected_dependent_variable,
                                      filter_labels: bool = False) -> np:
    # Preprocessing of df for classification
    avg_df = preprocessing_functions.cleanup_avg_df(avg_patient_cohort, selected_features, selected_dependent_variable)

    # Optional: Manually select Labels to focus clustering on
    if filter_labels:
        # labels_to_keep: List = ['Age', 'ethnicity', 'insurance', 'mechvent', 'White Blood Cells', 'sepsis_flag', 'cancer_flag', 'gender']
        labels_to_keep = ['Age', 'gender']
    else:
        labels_to_keep = avg_df.columns.to_list()  # use this option if all labels wanted

    # Clean up df & transform to numpy
    avg_df_without_nan = avg_df.fillna(0)
    filtered_df = avg_df_without_nan[avg_df_without_nan.columns.intersection(labels_to_keep)]
    avg_np = filtered_df.to_numpy()

    return avg_np


def calculate_cluster_kmeans(avg_np: np.ndarray, cohort_title: str, n_clusters: int, verbose: bool = False):
    """
    k-means clustering: choose amount n_clusters to calculate k centroids for these clusters
    :param verbose: boolean for printing STATUS
    :param cohort_title: title for STATUS
    :param avg_np: data
    :param n_clusters: amount of k clusters
    :return: list of responding clusters in the same order as patients list
    """
    if verbose:
        print(f'STATUS: Calculating k-means on {cohort_title} for {n_clusters} clusters.')

    # Calculate KMeans
    kmeans_obj = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_             # todo: directly merge these labels back to the icustay_id -> one place. Not return of labels as list but series icustay_id | cluster_label

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
    else:
        sh_score = round(silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)

    return clustering_labels_list, sh_score


def calculate_cluster_DBSCAN(avg_np: np.ndarray, cohort_title: str, eps: float, min_samples: int, verbose: bool = False):
    """
    density Based Spatial Clustering of Applications with Noise. Instances in dense region get clustered.
    :param verbose: boolean for printing STATUS
    :param cohort_title: title for STATUS
    :param avg_np: data
    :param eps: radius of clustering regions
    :param min_samples: needed for minimum amount of neighbors in clustering regions
    :return: list of responding clusters in the same order as patients list
    """
    if verbose:
        print(f'STATUS: Calculating DBSCAN on {cohort_title} for {eps} epsilon with {min_samples} min_samples.')

    # Calculate DBSCAN
    clustering_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(avg_np)           # we could use weights per label to give imputed labels less weight?
    clustering_labels_list = clustering_obj.labels_

    # get sh_score
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_labels_list)) < 2:
        sh_score = 0
    else:
        sh_score = round(silhouette_score(avg_np, labels=clustering_labels_list, metric='euclidean', random_state=0), 2)

    return clustering_labels_list, sh_score


def plot_clusters_on_3D_pacmap(plot_title, pacmap_data_points, cluster_count, sh_score, coloring, save_to_file):
    color_map = cm.get_cmap('brg', cluster_count)       # old: tab20c

    fig = plt.figure()
    fig.tight_layout(h_pad=2, w_pad=2)
    plt.suptitle(f'{plot_title} for clusters: {cluster_count}, sh_score: {sh_score}')
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.scatter(pacmap_data_points[:, 0], pacmap_data_points[:, 1], pacmap_data_points[:, 2], cmap=color_map,
                c=coloring, s=0.7, label='Patient')

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=color_map, norm=matplotlib.colors.Normalize(vmin=min(coloring), vmax=max(coloring))), ax=ax1)
    cb.set_label('Clusters')
    cb.set_ticks(list(set(coloring)))
    plt.legend()

    if save_to_file:
        plt.savefig(
            f'./output/clustering/3D_clusters_kmeans_{plot_title.replace(" ", "_")}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()
    plt.close()


def plot_sh_score_kmeans(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable,
                            filter_labels: bool = False, save_to_file: bool = False):
    """
    This function displays the Silhouette Score curve. With this an optimal cluster count for k-means can be selected.
    :param avg_patient_cohort:
    :param cohort_title:
    :param selected_features:
    :param selected_dependent_variable:
    :param filter_labels:
    :param save_to_file:
    :return:
    """
    print("STATUS: Calculating Silhouette Scores for k-means.")
    # Get cleaned avg_np
    avg_np = transform_df_to_np_for_clustering(avg_patient_cohort,
                                               selected_features,
                                               selected_dependent_variable,
                                               filter_labels)
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
        plt.savefig(f'./output/clustering/{f"Silhouette Score for k-Means on {cohort_title}".replace(" ", "_")}.png',
                    bbox_inches="tight")
    plt.show()
    plt.close()

    return None


def plot_clusters_on_pacmap(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable,
                            selected_cluster_count: int, filter_labels: bool = False, save_to_file: bool = False):
    # Get cleaned avg_np
    avg_np = transform_df_to_np_for_clustering(avg_patient_cohort,
                                               selected_features,
                                               selected_dependent_variable,
                                               filter_labels)
    # PacMap needed for visualization
    pacmap_data_points, death_list = data_visualization.calculate_pacmap(avg_patient_cohort=avg_patient_cohort,
                                                                         cohort_title=cohort_title,
                                                                         selected_features=selected_features,
                                                                         selected_dependent_variable=selected_dependent_variable)

    # Plot the cluster with best sh_score
    k_means_list, sh_score = calculate_cluster_kmeans(avg_np, cohort_title, n_clusters=selected_cluster_count)
    plot_title = f"k_Means_clusters_{selected_cluster_count} for {cohort_title}"
    plot_clusters_on_3D_pacmap(plot_title=plot_title, pacmap_data_points=pacmap_data_points, cluster_count=selected_cluster_count,
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
