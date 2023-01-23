import datetime

import matplotlib
import pacmap
from matplotlib import pyplot as plt


def display_pacmap(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable, save_to_file):
    # Preprocessing for classification
    avg_df = avg_patient_cohort.copy()
    avg_df = avg_df.drop(columns=['ethnicity',
                                  'insurance'])  # todo: these features can only be used if numeric - any good way to include?
    selected_features_final = selected_features
    selected_features_final.remove('icustay_id')
    selected_features_final.remove('stroke_type')
    selected_features_final.remove('ethnicity')
    selected_features_final.remove('insurance')
    selected_features_final.append(selected_dependent_variable)

    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                           'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_df = avg_df.drop(columns=available_dependent_variables)
    avg_df = avg_df[selected_features_final]
    avg_df = avg_df.fillna(0)

    # Filter and transform df to avg_np
    avg_np = avg_df.to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    print(f'STATUS: Conducting PaCMAP on {cohort_title}')
    embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, verbose=False, random_state=1)  # n_components = dimensions
    data_transformed = embedding.fit_transform(avg_np, init="pca")

    # Get death_list for markings
    death_list = []
    for v_icustay_id in avg_patient_cohort['icustay_id'].tolist():
        if avg_patient_cohort.loc[avg_patient_cohort['icustay_id'] == v_icustay_id, selected_dependent_variable].sum() > 0:
            death_list.append(1)
        else:
            death_list.append(0)

    # Plot PacMap
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title(f'PaCMAP visualization of {cohort_title} with {selected_dependent_variable}')
    ax1.scatter(data_transformed[:, 0],
                data_transformed[:, 1],
                data_transformed[:, 2],           # todo: this is only for 3D ? it seems like the z-dimension is missing
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
            f'./output/data_visualization/PacMap_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()
    plt.close()

    return None
