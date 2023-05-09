import os

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sn
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, _plot_dendrogram

from step_3_data_analysis.clustering import plot_sh_score, plot_k_means_on_pacmap, plot_k_prot_on_pacmap, \
    plot_sh_score_DBSCAN, plot_DBSCAN_on_pacmap, plot_SLINK_on_pacmap
from web_app.util import get_avg_cohort_cache


def clustering_page():
    ## Start of Page: User Input Selector
    st.markdown("<h2 style='text-align: left; color: black;'>Clustering</h2>", unsafe_allow_html=True)
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

        ## Select Clustering Specific Parameters
        # TODO: maybe put two Clustering methods next to each other for comparison
        col5, col6, col7, col8 = st.columns((0.25, 0.25, 0.25, 0.25))
        ALL_CLUSTERING_METHODS: list = ['kmeans', 'kprototype', 'DBSCAN', 'SLINK']
        clustering_method = col5.selectbox(label='Select clustering method', options=ALL_CLUSTERING_METHODS)
        ALL_CRITERIA: list = ['distance', 'maxclust', 'monocrit', 'inconsistent']
        if clustering_method == 'kmeans' or clustering_method == 'kprototype':
            selected_cluster_count = col6.number_input(label='Select cluster count k', min_value=1, max_value=20, value=2) # , format=None)
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'DBSCAN':
            selected_cluster_count = None
            selected_eps = col6.number_input(label='Select epsilon', min_value=0.01, max_value=10.00, value=0.51) # , format=None)
            selected_min_sample = col7.number_input(label='Select min_sample', min_value=1, max_value=100, value=5) # , format=None)
            selected_criterion = None
            selected_threshold = None
        elif clustering_method == 'SLINK':
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = col6.selectbox(label='Select separation criterion', options=ALL_CRITERIA)
            selected_threshold = col7.number_input(label='Select threshold t', min_value=0.01, max_value=2.00, value=1.00) # , format=None)
        else:
            selected_cluster_count = None
            selected_eps = None
            selected_min_sample = None
            selected_criterion = None
            selected_threshold = None
        st.markdown('___')

        ## Display Plots
        col1, col2 = st.columns((0.5, 0.5))
        if clustering_method == 'SLINK':
            col1.markdown("<h2 style='text-align: left; color: black;'>SLINK Dendrogram</h2>", unsafe_allow_html=True)
        else:
            col1.markdown("<h2 style='text-align: left; color: black;'>Silhouette Coefficient</h2>", unsafe_allow_html=True)
        col2.markdown("<h2 style='text-align: left; color: black;'>Cluster Visualization</h2>", unsafe_allow_html=True)

        if clustering_method == 'kmeans':
            # SH Score
            sh_score_plot = plot_sh_score(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          use_encoding=True, clustering_method='kmeans', save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            # Clustering
            clustering_plot = plot_k_means_on_pacmap(use_this_function=True,
                                                 display_sh_score=False,
                                                 selected_cohort=selected_cohort,
                                                 cohort_title=cohort_title,
                                                 use_case_name='frontend',
                                                 features_df=FEATURES_DF,
                                                 selected_features=selected_features,
                                                 selected_dependent_variable=selected_variable,
                                                 selected_cluster_count=selected_cluster_count,
                                                 use_encoding=True,
                                                 save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)

        elif clustering_method == 'kprototype':
            col1.write(f'Calculating the silhouette scores of kprototype for the first time can take up to 1 minute.')
            # SH Score
            sh_score_plot = plot_sh_score(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          use_encoding=False,        # not needed for kprot
                                          clustering_method='kprot', save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            # Clustering
            clustering_plot = plot_k_prot_on_pacmap(use_this_function=True,
                                                 display_sh_score=False,
                                                 selected_cohort=selected_cohort,
                                                 cohort_title=cohort_title,
                                                 use_case_name='frontend',
                                                 features_df=FEATURES_DF,
                                                 selected_features=selected_features,
                                                 selected_dependent_variable=selected_variable,
                                                 selected_cluster_count=selected_cluster_count,
                                                 use_encoding=False,        # not needed for kprot
                                                 save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)

        elif clustering_method == 'DBSCAN':
            col1.write(f'Iteratively change the DBSCAN parameters of epsilon and min_samples to optimize the clustering.')
            # SH Score
            sh_score_plot = plot_sh_score_DBSCAN(use_this_function=True, selected_cohort=selected_cohort,
                                          cohort_title=cohort_title, use_case_name='frontend', features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          selected_eps=selected_eps,
                                          selected_min_sample=selected_min_sample,
                                          use_encoding=True,
                                          save_to_file=False)

            col1.pyplot(sh_score_plot, use_container_width=True)
            # Clustering
            clustering_plot, dbscan_list = plot_DBSCAN_on_pacmap(use_this_function=True,
                                                display_sh_score=False,
                                                selected_cohort=selected_cohort,
                                                cohort_title=cohort_title,
                                                use_case_name='frontend',
                                                features_df=FEATURES_DF,
                                                selected_features=selected_features,
                                                selected_dependent_variable=selected_variable,
                                                selected_eps=selected_eps,
                                                selected_min_sample=selected_min_sample,
                                                use_encoding=True,  # not needed for kprot
                                                save_to_file=False)
            col2.pyplot(clustering_plot, use_container_width=True)

            if len(list(set(dbscan_list))) > 20:
                col2.warning('Warning: DBSCAN results in more than 20 clusters. Clustering might not be useful.')

        elif clustering_method == 'SLINK':
            col1.write(f'Iteratively change the threshold parameter to optimize the clustering.')
            # Get SLINK
            clustering_plot, clusters_list, slink_z = plot_SLINK_on_pacmap(use_this_function=True,
                                                                display_sh_score=False,
                                                                selected_cohort=selected_cohort,
                                                                cohort_title=cohort_title,
                                                                use_case_name='frontend',
                                                                features_df=FEATURES_DF,
                                                                selected_features=selected_features,
                                                                selected_dependent_variable=selected_variable,
                                                                use_encoding=True,
                                                                show_dendrogram=False,
                                                                separation_criterion=selected_criterion,
                                                                threshold=selected_threshold,
                                                                save_to_file=False)

            # Dendrogram
            # dendrogram_dict = dendrogram(slink_z)

            # plot = _plot_dendrogram(icoords=dendrogram_dict['icoords'],
            #                         dcoords=dendrogram_dict['dcoords'],
            #                         ivl,
            #                         p,
            #                         n,
            #                         mh,
            #                         orientation='top',
            #                         no_labels,
            #                         color_list,
            #                         leaf_font_size=None,
            #                         leaf_rotation=None,
            #                         contraction_marks=None,
            #                         ax=None,
            #                         above_threshold_color='C0')
            #
            # col1.pyplot(dendrogram_plot, use_container_width=True)

            # Plot Clustering
            col2.pyplot(clustering_plot, use_container_width=True)
            if len(list(set(clusters_list))) > 20:
                col2.warning('Warning: SLINK results in more than 20 clusters. Clustering might not be useful.')

        else:
            st.warning('Please select a clustering option.')
