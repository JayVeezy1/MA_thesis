import os

import pandas as pd
import streamlit as st

from step_5_fairness.fairness_analysis import get_fairness_report, plot_radar_fairness
from web_app.util import get_avg_cohort_cache, add_download_button, get_unfactorized_values, get_default_values, \
    insert_feature_selectors, add_single_feature_filter


def fairness_page():
    ## Start of Page: User Input Selector
    st.markdown("<h1 style='text-align: left; color: black;'>Fairness Analysis</h1>", unsafe_allow_html=True)
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
        # Feature Selector
        ALL_FEATURES = list(selected_cohort.columns)
        selected_features = insert_feature_selectors(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable)
        st.markdown('___')

        ## Fairness Selectors (with filter features selector)
        st.markdown("<h2 style='text-align: left; color: black;'>Feature Selection</h2>",
                    unsafe_allow_html=True)
        st.write('Select protected features/attributes and the related privileged values/classes.')
        selected_features_for_fairness, selected_privileged_values = add_single_feature_filter(selected_cohort, selected_features)
        st.markdown('___')

        ## Select Classification Specific Parameters
        # Selection 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        col1.markdown("<h2 style='text-align: left; color: black;'>Classification Method 1</h2>",
                      unsafe_allow_html=True)
        col3, col4, col6 = col1.columns((0.3, 0.3, 0.3))
        ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost']      # 'deeplearning_sequential'] not yet possible, needs rework in code
        classification_method = col3.selectbox(label='Select classification method 1',
                                               options=ALL_CLASSIFICATION_METHODS)
        ALL_SAMPLING_METHODS = ['no_sampling', 'oversampling']  # undersampling not useful
        sampling_method = col4.selectbox(label='Select sampling method 1', options=ALL_SAMPLING_METHODS)
        size_options = [0.20, 0.15, 0.25, 0.30, 0.35, 0.40]
        test_size1 = col6.selectbox(label='Select test data percentage 1', options=size_options)

        if classification_method == 'RandomForest_with_gridsearch':
            use_grid_search_1 = True
        else:
            use_grid_search_1 = False
        # Selection 2
        col5.markdown("<h2 style='text-align: left; color: black;'>Classification Method  2</h2>",
                      unsafe_allow_html=True)
        # Select Classification Specific Parameters
        col3, col4, col6 = col5.columns((0.3, 0.3, 0.3))
        classification_method_2 = col3.selectbox(label='Select classification method 2',
                                                 options=ALL_CLASSIFICATION_METHODS)
        sampling_method_2 = col4.selectbox(label='Select sampling method 2', options=ALL_SAMPLING_METHODS)
        test_size2 = col6.selectbox(label='Select test data percentage 2', options=size_options)

        if classification_method == 'RandomForest_with_gridsearch':
            use_grid_search_2 = True
        else:
            use_grid_search_2 = False
        st.markdown('___')


        ## Plot Fairness Metrics
        try:
            if len(selected_features_for_fairness) > 0:
                if len(selected_privileged_values) > 0:
                    fairness_report, metrics_plot, metrics_per_group_df, attributes_string = get_fairness_report(use_this_function=True,
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
                                                                                           test_size=test_size1,
                                                                                           use_grid_search=use_grid_search_1,
                                                                                           verbose=False,
                                                                                           protected_features=selected_features_for_fairness,
                                                                                           privileged_values=selected_privileged_values)
                    fairness_report_2, metrics_plot_2, metrics_per_group_df_2, attributes_string_2 = get_fairness_report(use_this_function=True,
                                                                                           selected_cohort=selected_cohort,
                                                                                           cohort_title=cohort_title,
                                                                                           features_df=FEATURES_DF,
                                                                                           selected_features=selected_features,
                                                                                           selected_dependent_variable=selected_variable,
                                                                                           classification_method=classification_method_2,
                                                                                           sampling_method=sampling_method_2,
                                                                                           use_case_name='frontend',
                                                                                           save_to_file=False,
                                                                                           plot_performance_metrics=True,
                                                                                           use_grid_search=use_grid_search_2,
                                                                                           test_size=test_size2,
                                                                                           verbose=False,
                                                                                           protected_features=selected_features_for_fairness,
                                                                                           privileged_values=selected_privileged_values)

                    # Plot Fairness
                    st.markdown("<h2 style='text-align: left; color: black;'>Fairness Metrics</h2>", unsafe_allow_html=True)
                    # Warning if values not reliable
                    col1, col2 = st.columns((0.5, 0.5))
                    recall_privileged_1 = metrics_per_group_df.transpose().loc[1, 'recall']
                    precision_privileged_1 = metrics_per_group_df.transpose().loc[1, 'precision']
                    if recall_privileged_1 == 0 or recall_privileged_1 == 1 or precision_privileged_1 == 0 or precision_privileged_1 == 1:
                        col1.write('Warning: Recall and Precision values can not be calculated reliably. '
                                   'True Positives may be 0 because no enough cases available for classification. '
                                   'It is recommended to select a larger privileged group.')
                    recall_privileged_2 = metrics_per_group_df_2.transpose().loc[1, 'recall']
                    precision_privileged_2 = metrics_per_group_df_2.transpose().loc[1, 'precision']
                    if recall_privileged_2 == 0 or recall_privileged_1 == 1 or precision_privileged_2 == 0 or precision_privileged_2 == 1:
                        col2.write('Warning: Recall and Precision values can not be calculated reliably. '
                                     'True Positives may be 0 because no enough cases available for classification. '
                                     'It is recommended to select a larger privileged group.')

                    # Plot Fairness Report 1
                    col1, col2, col3 = st.columns((0.2, 0.6, 0.2))
                    col1.dataframe(fairness_report)
                    add_download_button(position=col1, dataframe=fairness_report, title='fairness_report',
                                        cohort_title=cohort_title, keep_index=True)

                    # Plot Fairness Radar combined
                    categories = fairness_report_2.index.values.tolist()[1:]
                    result_1 = fairness_report['Values'].to_list()[1:]
                    result_2 = fairness_report_2['Values'].to_list()[1:]
                    fairness_radar_2 = plot_radar_fairness(categories=categories, list_of_results=[result_1, result_2])
                    col2.plotly_chart(figure_or_data=fairness_radar_2, use_container_width=True)

                    # Plot Fairness Report 2
                    col3.dataframe(fairness_report_2)
                    add_download_button(position=col3, dataframe=fairness_report_2, title='fairness_report_2',
                                        cohort_title=cohort_title, keep_index=True)
                    st.markdown('___')



                    # Plot Subgroups comparison 1
                    st.markdown("<h2 style='text-align: left; color: black;'>Subgroup Comparison</h2>", unsafe_allow_html=True)
                    col1, col_center, col2 = st.columns((0.475, 0.05, 0.475))
                    col1.dataframe(metrics_per_group_df.transpose(), use_container_width=True)
                    add_download_button(position=col1, dataframe=metrics_per_group_df, title='metrics_per_group_df',
                                        cohort_title=cohort_title, keep_index=True)
                    col1.pyplot(metrics_plot)
                    col1.write('Class 1 is made up of the selected protected features and their privileged attributes.')

                    # Plot Subgroups comparison 2
                    col2.dataframe(metrics_per_group_df_2.transpose(), use_container_width=True)
                    add_download_button(position=col2, dataframe=metrics_per_group_df_2, title='metrics_per_group_df_2',
                                        cohort_title=cohort_title, keep_index=True)
                    col2.pyplot(metrics_plot_2)
                else:
                    st.write('Select a privileged class to conduct a Fairness Analysis.')
            else:
                st.write('Select at least one protected attribute to conduct a Fairness Analysis.')

        except AttributeError:
            st.warning('Select different protected attributes to conduct a Fairness Analysis.')
        st.markdown('___')
