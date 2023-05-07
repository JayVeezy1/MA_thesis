import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sn
from matplotlib import pyplot as plt

from web_app.util import get_avg_cohort_cache


def clustering_page():
    ## Start of Page: User Input Selector
    st.markdown("<h2 style='text-align: left; color: black;'>Clustering</h2>", unsafe_allow_html=True)
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

        ## Select Clustering Specific Parameters
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLUSTERING_METHODS: list = ['kmeans', 'kprototype', 'DBSCAN']
        clustering_method = col5.selectbox(label='Select clustering method', options=ALL_CLUSTERING_METHODS)

        if clustering_method == 'kmeans':
            st.write('kmeans')
        elif clustering_method == 'kprototype':
            st.write('kprototype')
        elif clustering_method == 'DBSCAN':
            st.write('DBSCAN')
        else:
            st.warning('Please select a clustering method.')

