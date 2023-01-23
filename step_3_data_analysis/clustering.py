import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_k_means_on_cohort(avg_patient_cohort,  cohort_title, selected_features, selected_dependent_variable,
                                additional_options_title: str = None,
                                filter_labels: bool = False, save_to_file: bool = False):
    print("STATUS: Starting with k-means")

    # Preprocessing for classification
    avg_df = avg_patient_cohort.copy()
    # avg_df = avg_df.drop(columns=['ethnicity', 'insurance'])  # these features can only be used if numeric?
    selected_features_final = selected_features
    # selected_features_final.remove('ethnicity')
    # selected_features_final.remove('insurance')
    selected_features_final.append(selected_dependent_variable)

    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days', 'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_df = avg_df.drop(columns=available_dependent_variables)

    # PacMap needed for visualization
    pacmap_data, pacmap_patient_ids = calculate_pacmap_on_avg_df(avg_df.transpose())            # todo: really transpose?

    # Get the selected death label as death_list
    patient_ids = avg_df.index.to_list()
    death_list = []
    for patient in avg_df:
        if patient[selected_dependent_variable].sum() > 0:
            death_list.append(1)
        else:
            death_list.append(0)

    death_series = Series(death_list)
    death_series.index = patient_ids            # todo: is that correct?

    # fix NaN problem
    avg_df_without_nan = avg_df.fillna(0)

    # todo: CARRY ON HERE

    # Optional: Select Labels to Focus on
    if filter_labels:
        # you can select different labels here
        # labels_to_keep: List = ["Temp", "HR", "Resp", "pH", "Age", "Gender", "ICULOS", "SepsisLabel"]
        labels_to_keep = ["HR", "Resp", "ICULOS"]
    else:
        labels_to_keep = avg_df_without_nan.columns.to_list()  # use this option if all labels wanted
    filtered_df = avg_df_without_nan[avg_df_without_nan.columns.intersection(labels_to_keep)]
    print("Filtered df:", filtered_df)
    number_of_labels: int = len(filtered_df.columns)
    # Transform filtered_df to numpy
    avg_np = filtered_df.to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)

    # Plot silhouettes score analysis for k-means clustering
    print("\nSilhouettes Score Analysis: ")
    krange = list(range(2, 15))
    avg_silhouettes = []
    best_score = 0
    for n in krange:
        k_means_list, sh_score = calculate_cluster_kmeans(avg_np, n_clusters=n)
        avg_silhouettes.append(sh_score)
        if sh_score > best_score:
            best_score = sh_score
        if additional_options_title is None:
            title = f"k_Means_clusters_{n} for {training_set.name}"
        else:
            title = f"k_Means_clusters_{n} for {training_set.name} settings {additional_options_title}, feature_count {number_of_labels}"
        plot_clustering_with_silhouette_score_sepsis(plot_title=title, data=pacmap_data, sh_score=sh_score,
                                                     coloring=k_means_list,
                                                     patient_ids=patient_ids, training_set=training_set,
                                                     color_map='tab20c', save_to_file=save_to_file)

    plot_sh_scores(avg_silhouettes, cluster_range=krange, save_to_file=save_to_file, title="Silhouette Score for k-Means")


def calculate_pacmap_on_avg_df(avg_df, dimension: int = 2):
    """
    Calculate a PaCMAP transformation of the given avg_df.

    Based on TrainingSet.get_average_df()
    :param avg_df:
    :param dimension: dimension of the resulting PaCMAP mapping
    :return: data as returned by the PaCMAP algorithm
    """
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    embedding = pacmap.PaCMAP(n_dims=dimension, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=True, random_state=1)
    data_transformed = embedding.fit_transform(avg_np, init="pca")

    patient_ids = avg_df.columns.tolist()

    return data_transformed, patient_ids



def calculate_cluster_kmeans(avg_np: np.ndarray, n_clusters: int):
    """
    k-means clustering: choose amount n_clusters to calculate k centroids for these clusters

    :param avg_np:
    :param n_clusters: amount of k clusters
    :return: list of responding clusters in the same order as patients list
    """

    # use weights per label to give imputed labels less weight?
    kmeans_obj = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350,
                        verbose=True).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_

    sh_score = calculate_silhouette_score(avg_np, clustering_labels_list)

    return clustering_labels_list, sh_score


def calculate_silhouette_score(avg_np: np.ndarray, clustering_list: list) -> float:
    """
    Calculates the silhouette score, a quality measurement of clustering, for a given set of data and its clustering.
    :param avg_np: transposed numpy array of training_set.get_average_df()
    :param clustering_list:
    :return:
    """
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_list)) < 2:
        return 0
    else:
        return silhouette_score(avg_np, labels=clustering_list, metric='euclidean', random_state=0)


def plot_sh_scores(avg_silhouettes, cluster_range = None, eps_range = None, title="Silhouettes Score", save_to_file: bool = False):
    plt.figure(dpi=100)
    plt.title(title)
    if cluster_range is not None:
        plt.plot(cluster_range, avg_silhouettes)
        plt.xlabel("$k$")
    else:
        plt.plot(eps_range, avg_silhouettes)
        plt.xlabel("$eps$")
    plt.ylabel("Average Silhouettes Score")

    # Save & Show Plot
    plt.show()
    plot_title = 'Test1'
    plt.savefig(f'./output/clustering/{plot_title.replace(" ", "_")}.png', bbox_inches="tight")

    plt.close()
