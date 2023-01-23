import datetime
import pandas as pd
from matplotlib import pyplot as plt


def calculate_correlations_on_cohort(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable, save_to_file):
    # Preprocessing for correlation
    avg_patient_cohort = avg_patient_cohort.drop(columns=['icustay_id', 'stroke_type'])   # these features are needed for clustering, not correlations
    selected_features_corr = selected_features
    selected_features_corr.remove('icustay_id')
    selected_features_corr.remove('stroke_type')
    selected_features_corr.append(selected_dependent_variable)

    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days', 'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_patient_cohort = avg_patient_cohort.drop(columns=available_dependent_variables)

    # todo: check/ask if there is better option? How to use non-numeric columns for correlation? Not really useful? But still needed for clustering right?
    avg_patient_cohort[['insurance', 'ethnicity']] = avg_patient_cohort[['insurance', 'ethnicity']].apply(lambda x: pd.factorize(x)[0])
    # insurance: medicare = 0, mediaid = 1, Government = 2, private = 3, Self Pay = 4
    # ethnicity: WHITE = 0, UNKNOWN/NOT SPECIFIED = 1, HISPANIC OR LATINO = 2, BLACK = 3, OTHER = 4, ASIAN = 5

    # Calculate correlation
    avg_patient_cohort_corr = avg_patient_cohort[selected_features_corr].corr(numeric_only=False)
    death_corr = avg_patient_cohort_corr[selected_dependent_variable]
    sorted_death_corr = death_corr.sort_values(ascending=False)

    # Plot of correlation
    plot_death_corr = sorted_death_corr.drop(selected_dependent_variable)
    fig, ax1 = plt.subplots()
    color = plt.get_cmap('RdYlGn_r')
    max_value = plot_death_corr.max()
    min_value = plot_death_corr.min()
    y_axis = max(max_value, min_value) + 0.05
    if y_axis < 0.4:
        y_axis = 0.4
    ax1.bar([i for i in range(len(plot_death_corr))],
            [value for value in plot_death_corr],
            label=plot_death_corr.index.tolist(),
            color=[color((value - min_value) / 0.5) for value in plot_death_corr])          # (max_value + 0.05 - min_value))
    ax1.set_xticks([i for i in range(len(plot_death_corr))])
    ax1.set_xticklabels(plot_death_corr.index.tolist())
    ax1.set_ylim([-y_axis, y_axis])                       # correlation over 0.4 is not expected
    ax1.set_title(f"Correlation to {selected_dependent_variable} for {cohort_title}")
    plt.xticks(rotation=90)
    fig.tight_layout()
    if save_to_file:
        plt.savefig(f'./output/correlations/correlation_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    # todo: include these plots
    """
    # # Heatmap über alle Labels
    # # print("Heatmap:")
    fig, ax2 = py_plt.subplots()
    sb.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
               cmap='bwr', yticklabels=feature_names, xticklabels=feature_names, ax=ax2)
    ax2.set_title(f"Correlations in {training_set.name}, fixed values={fix_missing_values}, "
                  f"used interpolation={use_interpolation}")
    fig.tight_layout()
    plt.show()

    # Pairplot von ausgewählten Labels zu Sepsis und zueinander
    important_features = sorted_sepsis_corr.index[:3].tolist()
    important_features.extend(sorted_sepsis_corr.index[-3:].tolist())
    selected_labels_df = avg_df.transpose().filter(important_features, axis=1)
    avg_df_small = selected_labels_df.iloc[:100]  # scatter plot nur 100 patients
    sb.set_style('darkgrid')
    pairplot = sb.pairplot(avg_df_small)
    plt.show()
    """

    print('STATUS: calculate_correlations_on_cohort executed.')
    return None
