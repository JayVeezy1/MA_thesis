import os

import pandas as pd
import streamlit as st

from step_1_setup_data.cache_IO import load_data_from_cache
from step_3_data_analysis.correlations import plot_correlations
from step_3_data_analysis.general_statistics import calculate_deaths_table, calculate_feature_overview_table
from web_app.util import get_avg_cohort_cache, add_download_button


def data_analysis_page():
    ## Start of Page: User Input Selector
    st.markdown("<h1 style='text-align: left; color: black;'>General Data Analysis</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns((0.25, 0.25, 0.25))
    ALL_DEPENDENT_VARIABLES: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days', 'death_365_days']
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
        selected_features = st.multiselect(label='Select features', options=ALL_FEATURES, default=default_values)
        st.markdown('___')

        ## General Statistics DF
        st.markdown("<h2 style='text-align: left; color: black;'>Features Overview Table</h2>", unsafe_allow_html=True)
        overview_table = calculate_feature_overview_table(use_this_function=True,  # True | False
                                                          selected_cohort=selected_cohort,
                                                          features_df=FEATURES_DF,
                                                          selected_features=selected_features,
                                                          cohort_title=cohort_title,
                                                          use_case_name='frontend',
                                                          selected_dependent_variable=selected_variable,
                                                          save_to_file=False)
        # CSS to inject markdown, this removes index column from table
        hide_table_row_index = """ <style>
                                   thead tr th:first-child {display:none}
                                   tbody th {display:none}
                                   </style> """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(data=overview_table)
        add_download_button(position=None, dataframe=overview_table, title='overview_table', cohort_title=cohort_title)
        st.markdown('___')

        ## Deaths DF
        deaths_df = calculate_deaths_table(use_this_function=True,
                               use_case_name='frontend',
                               cohort_title=cohort_title,
                               selected_cohort=selected_cohort,
                               save_to_file=False)
        deaths_df = deaths_df.reset_index(drop=True)
        st.markdown("<h2 style='text-align: left; color: black;'>Death Cases Dataframe</h2>", unsafe_allow_html=True)
        st.dataframe(data=deaths_df.set_index(deaths_df.columns[0]), use_container_width=True)
        add_download_button(position=None, dataframe=deaths_df, title='deaths_df', cohort_title=cohort_title)
        st.markdown('___')

        ## Correlation
        st.markdown("<h2 style='text-align: left; color: black;'>Correlation</h2>", unsafe_allow_html=True)
        correlation_plot = plot_correlations(use_this_function=True,  # True | False
                                       use_plot_heatmap=False,
                                       use_plot_pairplot=False,
                                       cohort_title=cohort_title,
                                       selected_cohort=selected_cohort,
                                       features_df=FEATURES_DF,
                                       selected_features=selected_features,
                                       selected_dependent_variable=selected_variable,
                                       use_case_name='frontend',
                                       save_to_file=False)
        col1, col2, col3 = st.columns((0.4, 0.3, 0.3))
        col1.pyplot(correlation_plot, use_container_width=True)

        st.markdown('___')

        # TODO: add visualization (Pacmap or 3-feature-selection-plot) or simply inside Clustering?

