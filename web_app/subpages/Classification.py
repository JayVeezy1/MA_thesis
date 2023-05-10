import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sn
from matplotlib import pyplot as plt

from step_4_classification.classification import get_classification_report, get_confusion_matrix
from step_4_classification.classification_deeplearning import get_classification_report_deeplearning, \
    get_DL_confusion_matrix
from web_app.util import get_avg_cohort_cache, add_download_button


def classification_page():
    ## Start of Page: User Input Selector
    st.markdown("<h1 style='text-align: left; color: black;'>Classification</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns((0.25, 0.25, 0.25))
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
        default_values.insert(0, selected_variable)
        default_values.remove('age')            # remove these because too many categorical variables
        default_values.remove('stroke_type')
        selected_features = st.multiselect(label='Select features', options=ALL_FEATURES, default=default_values)

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
        st.markdown('___')

        ## CM and Report
        col1, col2, col3 = st.columns((0.3, 0.1, 0.4))
        col1.markdown("<h2 style='text-align: left; color: black;'>Confusion Matrix</h2>", unsafe_allow_html=True)
        col3.markdown("<h2 style='text-align: left; color: black;'>Classification Report</h2>", unsafe_allow_html=True)

        # TODO: maybe put two classification methods next to each other for comparison
        if classification_method == 'deeplearning_sequential':
            # TODO: check if DeepLearning cached correctly and also maybe make deeplearning parameters as user input?
            st.write(f'Calculating the classification with a deeplearning model for the first time takes about 1-2 minutes.')
            cm_df = get_DL_confusion_matrix(classification_method=classification_method,
                                          sampling_method=sampling_method,
                                          selected_cohort=selected_cohort,
                                          cohort_title=cohort_title,
                                          use_case_name='frontend',
                                          features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          verbose=False,
                                          save_to_file=False)
        else:
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
        col1.pyplot(fig1, use_container_width=True)


        # Get Report
        if classification_method == 'deeplearning_sequential':
            classification_report = get_classification_report_deeplearning(use_this_function=True,
                                                                          sampling_method=sampling_method,
                                                                          selected_cohort=selected_cohort,
                                                                          cohort_title=cohort_title,
                                                                          use_case_name='frontend',
                                                                          features_df=FEATURES_DF,
                                                                          selected_features=selected_features,
                                                                          selected_dependent_variable=selected_variable,
                                                                          verbose=False,
                                                                          save_to_file=False)

        else:
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
        col3.dataframe(classification_report)
        add_download_button(position=col3, dataframe=classification_report, title='classification_report', cohort_title=cohort_title)
        accuracy = round(classification_report.loc['accuracy', 'recall'], 2)
        recall = round(classification_report.loc['1.0', 'recall'], 2)
        precision = round(classification_report.loc['1.0', 'precision'], 2)

        col3.write(f'**Key metrics: Accuracy={accuracy}  |  Recall={recall}  |  Precision={precision}**')
        st.markdown('___')
