import os

import pandas as pd
import streamlit as st

from step_1_setup_data.cache_IO import load_data_from_cache
from step_3_data_analysis.correlations import plot_correlations, plot_pairplot
from step_3_data_analysis.general_statistics import calculate_deaths_table, calculate_feature_overview_table
from step_5_fairness.fairness_analysis import get_factorized_values
from web_app.util import get_avg_cohort_cache, add_download_button, get_default_values, insert_feature_selectors, \
    get_unfactorized_values, add_single_feature_filter


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
        # Feature Selector
        ALL_FEATURES = list(selected_cohort.columns)
        selected_features = insert_feature_selectors(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable)
        # Filter Selector
        filtered_cohort = selected_cohort
        checker_1 = st.checkbox(label='Optionally Filter the Dataset', value=False)
        if checker_1:
            st.write('Select categorical values to optionally filter the dataset for subgroups.')
            selected_filter_features, selected_filter_values = add_single_feature_filter(selected_cohort, selected_features)

            # Filter the cohort based on filter selections
            factorization_df = pd.read_excel('./supplements/FACTORIZATION_TABLE.xlsx')
            for i, filter_feature in enumerate(selected_filter_features):   # for-loop not optimal
                values = get_factorized_values(feature=filter_feature,
                                               privileged_values=selected_filter_values[i],
                                               factorization_df=factorization_df)
                if not len(values) < 1:         # no values selected, so no filtering
                    filtered_cohort = filtered_cohort[filtered_cohort[filter_feature].isin(values)]
        st.markdown('___')


        ## General Statistics DF
        st.markdown("<h2 style='text-align: left; color: black;'>Features Overview</h2>", unsafe_allow_html=True)
        overview_table = calculate_feature_overview_table(use_this_function=True,  # True | False
                                                          selected_cohort=filtered_cohort,      # todo: change this for all following
                                                          features_df=FEATURES_DF,
                                                          selected_features=selected_features,
                                                          cohort_title=cohort_title,
                                                          use_case_name='frontend',
                                                          selected_dependent_variable=selected_variable,
                                                          save_to_file=False)
        # # CSS to inject markdown, this removes index column from table
        # hide_table_row_index = """ <style>
        #                            thead tr th:first-child {display:none}
        #                            tbody th {display:none}
        #                            </style> """
        # st.markdown(hide_table_row_index, unsafe_allow_html=True)
        # st.table(data=overview_table)
        st.dataframe(data=overview_table.set_index(overview_table.columns[0]), use_container_width=True)
        add_download_button(position=None, dataframe=overview_table, title='overview_table', cohort_title=cohort_title, keep_index=False)
        st.markdown('___')

        ## Deaths DF
        deaths_df = calculate_deaths_table(use_this_function=True,
                               use_case_name='frontend',
                               cohort_title=cohort_title,
                               selected_cohort=filtered_cohort,
                               save_to_file=False)
        deaths_df = deaths_df.reset_index(drop=True)
        st.markdown("<h2 style='text-align: left; color: black;'>Mortality Overview</h2>", unsafe_allow_html=True)
        st.dataframe(data=deaths_df.set_index(deaths_df.columns[0]), use_container_width=True)
        add_download_button(position=None, dataframe=deaths_df, title='deaths_df', cohort_title=cohort_title, keep_index=False)
        st.markdown('___')

        ## Correlation
        correlation_plot = plot_correlations(use_this_function=True,  # True | False
                                       use_plot_heatmap=False,
                                       use_plot_pairplot=False,
                                       cohort_title=cohort_title,
                                       selected_cohort=filtered_cohort,
                                       features_df=FEATURES_DF,
                                       selected_features=selected_features,
                                       selected_dependent_variable=selected_variable,
                                       use_case_name='frontend',
                                       save_to_file=False)
        col1, col2, col3 = st.columns((0.6, 0.1, 0.4))
        col1.markdown("<h2 style='text-align: left; color: black;'>Correlation</h2>", unsafe_allow_html=True)
        col1.pyplot(correlation_plot, use_container_width=True)

        ## Pairplot of 2 Features
        col3.markdown("<h2 style='text-align: left; color: black;'>Feature Distribution</h2>", unsafe_allow_html=True)
        selected_features_pairplot = col3.multiselect(label='Select features', options=ALL_FEATURES, default=[selected_variable, 'oasis'], max_selections=3)
        pairplot = plot_pairplot(cohort_title=cohort_title,
                      selected_cohort=filtered_cohort,
                      features_df=FEATURES_DF,
                      selected_features=selected_features_pairplot,
                      selected_dependent_variable=selected_variable,
                      selected_patients=100,
                      use_case_name='frontend',
                      save_to_file=False)
        col3.pyplot(pairplot, use_container_width=True)

        st.markdown('___')
