import datetime

import matplotlib
import pacmap
from matplotlib import pyplot as plt

from step_2_preprocessing import preprocessing_functions


def calculate_pacmap(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable):
    """
    :return pacmap_data_points = data points, and death_list = markings
    """
    avg_df = preprocessing_functions.cleanup_avg_df(avg_patient_cohort, selected_features, selected_dependent_variable)

    # Filter and transform df to avg_np
    avg_np = avg_df.to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    print(f'STATUS: Conducting PaCMAP on {cohort_title}')
    embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, verbose=False, random_state=1)  # n_components = dimensions
    pacmap_data_points = embedding.fit_transform(avg_np, init="pca")

    # Get death_list for markings
    death_list = []
    for v_icustay_id in avg_patient_cohort['icustay_id'].tolist():
        if avg_patient_cohort.loc[avg_patient_cohort['icustay_id'] == v_icustay_id, selected_dependent_variable].sum() > 0:
            death_list.append(1)
        else:
            death_list.append(0)

    return pacmap_data_points, death_list         # pacmap_data_points = data points, death_list = markings


def display_pacmap(avg_patient_cohort, cohort_title, use_case_name, selected_features, selected_dependent_variable, save_to_file):
    pacmap_data_points, death_list = calculate_pacmap(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable)

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
