import os

import pandas as pd
import streamlit as st

from step_6_subgroup_analysis.subgroup_analysis import calculate_clusters_overview_table, \
    compare_classification_models_on_clusters
from web_app.util import get_avg_cohort_cache, add_download_button


def subgroup_analysis_page():
    ## Start of Page: User Input Selector
    st.markdown("<h1 style='text-align: left; color: black;'>Subgroups Analysis</h1>", unsafe_allow_html=True)
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

        ## Select Clustering Specific Parameters
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLUSTERING_METHODS: list = ['kmeans', 'kprototype', 'DBSCAN', 'SLINK']
        clustering_method = col5.selectbox(label='Select clustering method', options=ALL_CLUSTERING_METHODS)
        ALL_CRITERIA: list = ['maxclust', 'distance', 'monocrit', 'inconsistent']
        if clustering_method == 'kmeans' or clustering_method == 'kprototype':
            selected_cluster_count = col6.number_input(label='Select cluster count k', min_value=1, max_value=20,
                                                       value=3)  # , format=None)
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'DBSCAN':
            selected_cluster_count = None
            selected_eps = col6.number_input(label='Select epsilon', min_value=0.01, max_value=10.00,
                                             value=0.51)  # , format=None)
            selected_min_sample = col7.number_input(label='Select min_sample', min_value=1, max_value=100,
                                                    value=5)  # , format=None)
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'SLINK':
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = col6.selectbox(label='Select separation criterion', options=ALL_CRITERIA)
            selected_threshold = col7.number_input(label='Select threshold t', min_value=0.01, max_value=100.00,
                                                   value=1.00)  # , format=None)
        else:
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        st.markdown('___')

        if clustering_method == 'kmeans' or clustering_method == 'kprototype':
            st.markdown("<h2 style='text-align: left; color: black;'>Features per Cluster</h2>", unsafe_allow_html=True)
            st.write('The distribution of feature values can be used for subgroup detection.')
            selected_show_influences = st.checkbox('Show Value Influences in Features Table.')
            clusters_overview_table = calculate_clusters_overview_table(use_this_function=True,
                                                                       selected_cohort=selected_cohort,
                                                                       cohort_title=cohort_title,
                                                                       use_case_name='frontend',
                                                                       features_df=FEATURES_DF,
                                                                       selected_features=selected_features,
                                                                       selected_dependent_variable=selected_variable,
                                                                       selected_k_means_count=selected_cluster_count,
                                                                       show_value_influences=selected_show_influences,
                                                                       use_encoding=True,
                                                                       save_to_file=False)
            # hide_dataframe_row_index = """
            #             <style>
            #             .row_heading.level0 {display:none}
            #             .blank {display:none}
            #             </style>
            #             """
            # st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.dataframe(clusters_overview_table.set_index(clusters_overview_table.columns[0]), use_container_width=True)
            add_download_button(position=None, dataframe=clusters_overview_table, title='clusters_overview_table', cohort_title=cohort_title)
            st.markdown('___')


            ## Select Classification Specific Parameters
            st.markdown("<h2 style='text-align: left; color: black;'>Classification Metrics per Cluster</h2>", unsafe_allow_html=True)
            st.write('The comparison of prediction metrics across clusters can be an indicator for a subsequent fairness analysis.')
            col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
            ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost',
                                                'deeplearning_sequential']
            classification_method = col5.selectbox(label='Select classification method',
                                                   options=ALL_CLASSIFICATION_METHODS)
            ALL_SAMPLING_METHODS = ['no_sampling', 'oversampling']  # undersampling not useful
            sampling_method = col6.selectbox(label='Select classification method', options=ALL_SAMPLING_METHODS)
            if classification_method == 'RandomForest_with_gridsearch':
                use_grid_search = True
            else:
                use_grid_search = False
            # st.markdown('___')

            # Plot Classification per Cluster Table
            classification_overview_table = compare_classification_models_on_clusters(use_this_function=True,
                                                                                      use_case_name='frontend',
                                                                                      features_df=FEATURES_DF,
                                                                                      selected_features=selected_features,
                                                                                      selected_cohort =selected_cohort,
                                                                                      classification_method=classification_method,
                                                                                      sampling_method=sampling_method,
                                                                                      clustering_method=clustering_method,
                                                                                      cohort_title=cohort_title,
                                                                                      dependent_variable=selected_variable,
                                                                                      selected_k_means_count=selected_cluster_count,
                                                                                      check_sh_score=False,
                                                                                      use_grid_search=use_grid_search,
                                                                                      use_encoding=True,
                                                                                      save_to_file=False)
            st.dataframe(classification_overview_table, use_container_width=True)
            add_download_button(position=None, dataframe=classification_overview_table, title='classification_overview_table', cohort_title=cohort_title)
            st.markdown('___')

        else:
            st.warning('Clusters overview tables only available for clustering methods with preselected cluster count (kmeans and kprototype).')
            st.markdown('___')

