import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sn
from matplotlib import pyplot as plt

from step_4_classification.classification import get_classification_report, get_confusion_matrix
from web_app.util import get_avg_cohort_cache


def classification_page():
    ## Start of Page: User Input Selector
    st.markdown("<h2 style='text-align: left; color: black;'>Features Overview Table</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns((0.25, 0.25, 0.25, 0.25))
    ALL_DEPENDENT_VARIABLES: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                     'death_365_days']
    selected_variable = col1.selectbox(label='Select dependent variable', options=ALL_DEPENDENT_VARIABLES)
    ALL_DATABASES: list = ['complete', 'metavision', 'carevue']
    selected_database = col2.selectbox(label='Select database', options=ALL_DATABASES)
    ALL_STROKE_TYPES: list = ['all_stroke', 'ischemic', 'other_stroke', 'hemorrhagic']
    selected_stroke_type = col3.selectbox(label='Select stroke type', options=ALL_STROKE_TYPES)
    cohort_title = 'scaled_' + selected_database + '_avg_cohort_' + selected_stroke_type

    ## Get Cohort from streamlit cache function
    upload_filename = './web_app/data_upload/exports/frontend/avg_patient_cohort.csv'
    if not os.path.isfile(upload_filename):
        st.warning('Warning: No dataset was uploaded. Please, first upload a dataset at the "Data Upload" page.')
    else:
        PROJECT_PATH = './web_app/data_upload/'
        FEATURES_DF = pd.read_excel('./supplements/FEATURE_PREPROCESSING_TABLE.xlsx')

        selected_cohort = get_avg_cohort_cache(project_path=PROJECT_PATH,
                                             use_case_name='frontend',
                                             features_df=FEATURES_DF,
                                             selected_database=selected_database,
                                             selected_stroke_type=selected_stroke_type,
                                             delete_existing_cache=False,
                                             selected_patients=[])  # empty = all
        ALL_FEATURES = list(selected_cohort.columns)
        default_values = [x for x in ALL_FEATURES if x not in ALL_DEPENDENT_VARIABLES]
        selected_features = col4.multiselect(label='Select features', options=ALL_FEATURES, default=default_values)

        ## Select Classification Specific Parameters
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost', 'deeplearning_sequential']
        classification_method = col5.selectbox(label='Select classification method', options=ALL_CLASSIFICATION_METHODS)
        ALL_SAMPLING_METHODS = ['no_sampling', 'oversampling']  # undersampling not useful
        sampling_method = col6.selectbox(label='Select classification method', options=ALL_SAMPLING_METHODS)
        if classification_method == 'RandomForest_with_gridsearch':
            use_grid_search = True
        else:
            use_grid_search = False

        ## CM and Report
        st.markdown("<h2 style='text-align: left; color: black;'>Confusion Matrix and Classification Report</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns((0.5, 0.5))

        # TODO: check if DeepLearning cached correctly and maybe add warning "takes time", also maybe make deeplearning parameters user input?
        # TODO: maybe put two classification methods next to each other for comparison
        cm_df = get_confusion_matrix(use_this_function=True,  # True | False
                                      classification_method=classification_method,
                                      sampling_method=sampling_method,
                                      selected_cohort=selected_cohort,
                                      cohort_title=cohort_title,
                                      use_case_name='frontend',
                                      features_df=FEATURES_DF,
                                      selected_features=selected_features,
                                      selected_dependent_variable=selected_variable,
                                      use_grid_search=use_grid_search,
                                      verbose=False,
                                      save_to_file=False)
        # Transform cm_df to plot object
        cmap = 'viridis'
        fig1, ax1 = plt.subplots()
        # add totals to cm_df
        sum_col = []
        for c in cm_df.columns:
            sum_col.append(cm_df[c].sum())
        sum_lin = []
        for item_line in cm_df.iterrows():
            sum_lin.append(item_line[1].sum())
        cm_df['sum_actual'] = sum_lin
        sum_col.append(np.sum(sum_lin))
        cm_df.loc['sum_predicted'] = sum_col
        # create seaborn heatmap
        ax1 = sn.heatmap(
            data=cm_df,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 15},
            linewidths=0.5,
            ax=ax1,
            cbar=False,
            cmap=cmap,
            vmin=0,
            vmax=(cm_df['sum_actual']['sum_predicted'] + 20)
            # adding a bit to max value -> not such a strong color difference
        )
        # sn.set(font_scale=3.0)
        # set ticklabels rotation (0 rotation, but with this horizontal)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)
        # titles and legends
        plt.tick_params(axis='x', which='major', labelsize=11, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        ax1.set_title(f"{classification_method} on {cohort_title}, {sampling_method}", wrap=True)
        plt.tight_layout()

        # Get Report
        classification_report = get_classification_report(use_this_function=True,  # True | False
                                                          display_confusion_matrix=False,  # option for CM
                                                          classification_method=classification_method,
                                                          sampling_method=sampling_method,
                                                          selected_cohort=selected_cohort,
                                                          cohort_title=cohort_title,
                                                          use_case_name='frontend',
                                                          features_df=FEATURES_DF,
                                                          selected_features=selected_features,
                                                          selected_dependent_variable=selected_variable,
                                                          use_grid_search=use_grid_search,
                                                          verbose=False,
                                                          save_to_file=False)

        col1.pyplot(fig1, use_container_width=True)
        col2.dataframe(classification_report)
