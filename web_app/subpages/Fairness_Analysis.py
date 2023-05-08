import os

import pandas as pd
import streamlit as st

from step_5_fairness.fairness_analysis import get_fairness_report, create_performance_metrics_plot
from web_app.util import get_avg_cohort_cache


def fairness_page():
    ## Start of Page: User Input Selector
    st.markdown("<h2 style='text-align: left; color: black;'>Fairness Analysis</h2>", unsafe_allow_html=True)
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
        selected_features = st.multiselect(label='Select features', options=ALL_FEATURES, default=default_values)

        ## Select Classification Specific Parameters
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost',
                                            'deeplearning_sequential']
        classification_method = col5.selectbox(label='Select classification method', options=ALL_CLASSIFICATION_METHODS)
        ALL_SAMPLING_METHODS = ['no_sampling', 'oversampling']  # undersampling not useful
        sampling_method = col6.selectbox(label='Select classification method', options=ALL_SAMPLING_METHODS)
        if classification_method == 'RandomForest_with_gridsearch':
            use_grid_search = True
        else:
            use_grid_search = False
        st.markdown('___')


        ## Fairness Report and Performance Metrics Plot
        fairness_report, metrics_plot = get_fairness_report(use_this_function=True,
                                                  selected_cohort=selected_cohort,
                                                  cohort_title=cohort_title,
                                                  features_df=FEATURES_DF,
                                                  selected_features=selected_features,
                                                  selected_dependent_variable=selected_variable,
                                                  classification_method=classification_method,
                                                  sampling_method=sampling_method,
                                                  use_case_name='frontend',
                                                  save_to_file=False,
                                                  plot_performance_metrics=True,
                                                  use_grid_search=use_grid_search,
                                                  verbose=False)


        col1, col2 = st.columns((0.5, 0.5))
        col1.markdown("<h2 style='text-align: left; color: black;'>Fairness Report</h2>", unsafe_allow_html=True)
        col1.dataframe(fairness_report)

        col2.markdown("<h2 style='text-align: left; color: black;'>Subgroups Comparison</h2>", unsafe_allow_html=True)
        col2.pyplot(metrics_plot)
