import datetime
import matplotlib
import pacmap
from matplotlib import pyplot as plt


def preprocess_for_pacmap(selected_cohort, features_df, selected_features, selected_dependent_variable):
    # Preprocessing for Clustering: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)  # keeping selected_dependent_variable for clustering?
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    avg_cohort_without_nan = selected_cohort[selected_features].fillna(0)

    # print(f'CHECK: {len(selected_features)} features used for PacMap.')

    return avg_cohort_without_nan.to_numpy()


def calculate_pacmap(selected_cohort, cohort_title, features_df, selected_features, selected_dependent_variable):
    # Returns: pacmap_data_points = data points, and death_list = markings

    # Filter and transform df to avg_np
    avg_np = preprocess_for_pacmap(selected_cohort, features_df, selected_features, selected_dependent_variable)

    print(f'STATUS: Conducting PaCMAP on {cohort_title}')
    embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, verbose=False,
                              random_state=1)  # n_components = dimensions
    pacmap_data_points = embedding.fit_transform(avg_np, init="pca")

    # Get death_list for markings
    death_list = []
    for v_icustay_id in selected_cohort['icustay_id'].tolist():
        if selected_cohort.loc[selected_cohort['icustay_id'] == v_icustay_id, selected_dependent_variable].sum() > 0:
            death_list.append(1)
        else:
            death_list.append(0)

    return pacmap_data_points, death_list  # pacmap_data_points = data points, death_list = markings


def display_pacmap(use_this_function: False, selected_cohort, cohort_title, use_case_name, features_df, selected_features, selected_dependent_variable,
                   save_to_file):
    if not use_this_function:
        return None

    # Calculate PacMap
    pacmap_data_points, death_list = calculate_pacmap(selected_cohort, cohort_title, features_df, selected_features, selected_dependent_variable)

    # Plot PacMap
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title(f'PaCMAP visualization of {cohort_title} with {selected_dependent_variable}')
    ax1.scatter(pacmap_data_points[:, 0],
                pacmap_data_points[:, 1],
                pacmap_data_points[:, 2],
                cmap="cool",
                c=death_list,
                s=0.6, label="Patient")
    plt.legend()
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap="cool",
                                              norm=matplotlib.colors.Normalize(
                                                  vmin=min(death_list),
                                                  vmax=max(death_list))),
                 ax=ax1)

    if save_to_file:
        plt.savefig(
            f'./output/{use_case_name}/data_visualization/PacMap_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()
    plt.close()

    return None
