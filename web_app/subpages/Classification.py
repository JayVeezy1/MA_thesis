import os

from PIL import Image

import numpy as np
import pandas as pd
import shap
import streamlit as st
import seaborn as sn
from matplotlib import pyplot as plt

from step_4_classification.classification import get_classification_report, get_confusion_matrix, get_auc_score, \
    get_shapely_values
from step_4_classification.classification_deeplearning import get_classification_report_deeplearning, \
    get_DL_confusion_matrix
from web_app.util import get_avg_cohort_cache, add_download_button, get_default_values, get_preselection_one, \
    get_preselection_two, insert_feature_selectors


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

        # Feature Selector
        ALL_FEATURES = list(selected_cohort.columns)
        selected_features = insert_feature_selectors(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable)
        st.markdown('___')


        ## Select Classification Specific Parameters
        # Selection 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        col1.markdown("<h2 style='text-align: left; color: black;'>Classification Method 1</h2>",
                      unsafe_allow_html=True)
        col3, col4, col6 = col1.columns((0.3, 0.3, 0.3))
        ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost',
                                            'deeplearning_sequential']
        classification_method = col3.selectbox(label='Select classification method 1',
                                               options=ALL_CLASSIFICATION_METHODS)
        ALL_SAMPLING_METHODS = ['no_sampling', 'oversampling']  # undersampling not useful
        sampling_method = col4.selectbox(label='Select sampling method 1', options=ALL_SAMPLING_METHODS)
        size_options = [0.20, 0.15, 0.25, 0.30, 0.35, 0.40]
        test_size1 = col6.selectbox(label='Select test data percentage 1', options=size_options) # , default=0.20)

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
        test_size2 = col6.selectbox(label='Select test data percentage 2', options=size_options) # , default=0.20)

        if classification_method == 'RandomForest_with_gridsearch':
            use_grid_search_2 = True
        else:
            use_grid_search_2 = False
        st.markdown('___')

        # todo future work: maybe make deeplearning parameters as user input

        # Get Report Selection 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        if classification_method == 'deeplearning_sequential':
            classification_report = get_classification_report_deeplearning(use_this_function=True,
                                                                           sampling_method=sampling_method,
                                                                           selected_cohort=selected_cohort,
                                                                           cohort_title=cohort_title,
                                                                           use_case_name='frontend',
                                                                           features_df=FEATURES_DF,
                                                                           selected_features=selected_features,
                                                                           selected_dependent_variable=selected_variable,
                                                                           test_size=test_size1,
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
                                                              test_size=test_size1,
                                                              use_grid_search=use_grid_search_1,
                                                              verbose=False,
                                                              save_to_file=False)
        col1.markdown("<h2 style='text-align: left; color: black;'>Classification Report</h2>", unsafe_allow_html=True)
        col1.dataframe(classification_report, use_container_width=True)
        add_download_button(position=col1, dataframe=classification_report, title='classification_report',
                            cohort_title=cohort_title, keep_index=False)
        accuracy = round(classification_report.loc['accuracy', 'recall'], 2)
        recall = round(classification_report.loc['1.0', 'recall'], 2)
        precision = round(classification_report.loc['1.0', 'precision'], 2)
        col1.write(f'**Key metrics: Accuracy={accuracy}  |  Recall={recall}  |  Precision={precision}**')


        # Get Report Selection 2
        if classification_method_2 == 'deeplearning_sequential':
            classification_report_2 = get_classification_report_deeplearning(use_this_function=True,
                                                                             sampling_method=sampling_method_2,
                                                                             selected_cohort=selected_cohort,
                                                                             cohort_title=cohort_title,
                                                                             use_case_name='frontend',
                                                                             features_df=FEATURES_DF,
                                                                             selected_features=selected_features,
                                                                             selected_dependent_variable=selected_variable,
                                                                             test_size=test_size2,
                                                                             verbose=False,
                                                                             save_to_file=False)

        else:
            classification_report_2 = get_classification_report(use_this_function=True,  # True | False
                                                                display_confusion_matrix=False,  # option for CM
                                                                classification_method=classification_method_2,
                                                                sampling_method=sampling_method_2,
                                                                selected_cohort=selected_cohort,
                                                                cohort_title=cohort_title,
                                                                use_case_name='frontend',
                                                                features_df=FEATURES_DF,
                                                                selected_features=selected_features,
                                                                selected_dependent_variable=selected_variable,
                                                                test_size=test_size2,
                                                                use_grid_search=use_grid_search_2,
                                                                verbose=False,
                                                                save_to_file=False)
        # col5.markdown("<h2 style='text-align: left; color: black;'>Classification Report 2</h2>", unsafe_allow_html=True)
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')

        col5.dataframe(classification_report_2, use_container_width=True)
        add_download_button(position=col5, dataframe=classification_report_2, title='classification_report_2',
                            cohort_title=cohort_title, keep_index=False)
        accuracy = round(classification_report_2.loc['accuracy', 'recall'], 2)
        recall = round(classification_report_2.loc['1.0', 'recall'], 2)
        precision = round(classification_report_2.loc['1.0', 'precision'], 2)
        col5.write(f'**Key metrics: Accuracy={accuracy}  |  Recall={recall}  |  Precision={precision}**')
        st.markdown('___')


        ## CM Selection 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        col1.markdown("<h2 style='text-align: left; color: black;'>Confusion Matrix</h2>", unsafe_allow_html=True)
        if classification_method == 'deeplearning_sequential':
            st.write(f'Calculating the classification with a deeplearning model for the first time takes about 1-2 minutes.')
            cm_df = get_DL_confusion_matrix(classification_method=classification_method,
                                          sampling_method=sampling_method,
                                          selected_cohort=selected_cohort,
                                          cohort_title=cohort_title,
                                          use_case_name='frontend',
                                          features_df=FEATURES_DF,
                                          selected_features=selected_features,
                                          selected_dependent_variable=selected_variable,
                                          test_size=test_size1,
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
                                          test_size=test_size1,
                                          use_grid_search=use_grid_search_1,
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


        ## CM Selection 2
        # col5.markdown("<h2 style='text-align: left; color: black;'>Confusion Matrix 2</h2>", unsafe_allow_html=True)
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')

        if classification_method_2 == 'deeplearning_sequential':
            st.write(
                f'Calculating the classification with a deeplearning model for the first time takes about 1-2 minutes.')
            cm_df_2 = get_DL_confusion_matrix(classification_method=classification_method_2,
                                            sampling_method=sampling_method_2,
                                            selected_cohort=selected_cohort,
                                            cohort_title=cohort_title,
                                            use_case_name='frontend',
                                            features_df=FEATURES_DF,
                                            selected_features=selected_features,
                                            selected_dependent_variable=selected_variable,
                                            test_size=test_size2,
                                            verbose=False,
                                            save_to_file=False)
        else:
            cm_df_2 = get_confusion_matrix(use_this_function=True,  # True | False
                                         classification_method=classification_method_2,
                                         sampling_method=sampling_method_2,
                                         selected_cohort=selected_cohort,
                                         cohort_title=cohort_title,
                                         use_case_name='frontend',
                                         features_df=FEATURES_DF,
                                         selected_features=selected_features,
                                         selected_dependent_variable=selected_variable,
                                         test_size=test_size2,
                                         use_grid_search=use_grid_search_1,
                                         verbose=False,
                                         save_to_file=False)

        # Transform cm_df to plot object
        cmap = 'viridis'
        fig1, ax1 = plt.subplots()
        # add totals to cm_df
        sum_col = []
        for c in cm_df_2.columns:
            sum_col.append(cm_df_2[c].sum())
        sum_lin = []
        for item_line in cm_df_2.iterrows():
            sum_lin.append(item_line[1].sum())
        cm_df_2['sum_actual'] = sum_lin
        sum_col.append(np.sum(sum_lin))
        cm_df_2.loc['sum_predicted'] = sum_col
        # create seaborn heatmap
        ax1 = sn.heatmap(
            data=cm_df_2,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 15},
            linewidths=0.5,
            ax=ax1,
            cbar=False,
            cmap=cmap,
            vmin=0,
            vmax=(cm_df_2['sum_actual']['sum_predicted'] + 20)
            # adding a bit to max value -> not such a strong color difference
        )
        # sn.set(font_scale=3.0)
        # set ticklabels rotation (0 rotation, but with this horizontal)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=10)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)
        # titles and legends
        plt.tick_params(axis='x', which='major', labelsize=11, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        ax1.set_title(f"{classification_method_2} on {cohort_title}, {sampling_method_2}", wrap=True)
        plt.tight_layout()
        col5.pyplot(fig1, use_container_width=True)
        st.markdown('___')

        # AUROC 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        col1.markdown("<h2 style='text-align: left; color: black;'>AUROC</h2>", unsafe_allow_html=True)
        col1.write('Area under the receiver operating characteristic indicates a models performance.')
        if classification_method == 'deeplearning_sequential':
            col1.write(f'Calculating the classification with a deeplearning model for the first time takes about 1-2 minutes.')

        auc_score, auroc_plot, auc_prc_score, auc_prc_plot = get_auc_score(use_this_function=True,  # True | False
                                                                    classification_method=classification_method,
                                                                    sampling_method=sampling_method,
                                                                    selected_cohort=selected_cohort,
                                                                    cohort_title=cohort_title,
                                                                    use_case_name='frontend',
                                                                    features_df=FEATURES_DF,
                                                                    selected_features=selected_features,
                                                                    selected_dependent_variable=selected_variable,
                                                                    show_plot=False,
                                                                    test_size=test_size1,
                                                                    use_grid_search=use_grid_search_1,
                                                                    verbose=False,
                                                                    save_to_file=False)

       # Plot AUROC 1
        col1.pyplot(auroc_plot)

        # AUROC 2
        # col5.markdown("<h2 style='text-align: left; color: black;'>AUROC 2</h2>", unsafe_allow_html=True)
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')

        if classification_method_2 == 'deeplearning_sequential':
            col5.write(
                f'Calculating the classification with a deeplearning model for the first time takes about 1-2 minutes.')

        auc_score_2, auroc_plot_2, auc_prc_score_2, auc_prc_plot_2 = get_auc_score(use_this_function=True,
                                                                                   # True | False
                                                                                   classification_method=classification_method_2,
                                                                                   sampling_method=sampling_method_2,
                                                                                   selected_cohort=selected_cohort,
                                                                                   cohort_title=cohort_title,
                                                                                   use_case_name='frontend',
                                                                                   features_df=FEATURES_DF,
                                                                                   selected_features=selected_features,
                                                                                   selected_dependent_variable=selected_variable,
                                                                                   show_plot=False,
                                                                                   test_size=test_size2,
                                                                                   use_grid_search=use_grid_search_2,
                                                                                   verbose=False,
                                                                                   save_to_file=False)
        # Plot AUROC 2
        col5.pyplot(auroc_plot_2)
        st.markdown('___')


        # Plot AUPRC 1
        col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
        col1.markdown("<h2 style='text-align: left; color: black;'>AUPRC</h2>", unsafe_allow_html=True)
        col1.write('Area under the precision-recall curve is a helpful indicator when working with imbalanced data.')
        col1.pyplot(auc_prc_plot)

        # Plot AUPRC 2
        # col5.markdown("<h2 style='text-align: left; color: black;'>AUPRC</h2>", unsafe_allow_html=True)
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')
        col5.write('')

        col5.pyplot(auc_prc_plot_2)

        st.markdown('___')

        # Add Shapely Feature Relevance
        st.markdown("<h2 style='text-align: left; color: black;'>Feature Relevance</h2>", unsafe_allow_html=True)

        # TODO: own function needed for deeplearning?
        # if classification_method_2 == 'deeplearning_sequential':
        #     shap_df, shap_waterfall_plot, shap_beeswarm_plot, shap_partial_dependence_plot, shap_scatter_plot = get_shapely_relevance_deeplearning()
        # else:

        if 'oasis' in selected_features:
            selected_shap_feature = st.multiselect(label='Select a Feature for Shapley Analysis', options=selected_features, default='oasis', max_selections=1)
        else:
            selected_shap_feature = st.multiselect(label='Select a Feature for Shapley Analysis', options=selected_features, max_selections=1)

        # Calculate Shapleys if Button pressed
        if st.button('Start Shapley Calculation'):
            shap_values, sampling_title = get_shapely_values(use_this_function=True,  # True | False
                                                                selected_feature=selected_shap_feature,
                                                                classification_method=classification_method,
                                                                sampling_method=sampling_method,
                                                                selected_cohort=selected_cohort,
                                                                cohort_title=cohort_title,
                                                                use_case_name='frontend',
                                                                features_df=FEATURES_DF,
                                                                selected_features=selected_features,
                                                                selected_dependent_variable=selected_variable,
                                                                show_plot=False,
                                                                use_grid_search=use_grid_search_1,
                                                                test_size=test_size1,
                                                                verbose=False,
                                                                save_to_cache=True,
                                                                save_to_file=False)

            col1, col2, col5 = st.columns((0.475, 0.05, 0.475))
            # col1.dataframe(shap_values[:, selected_shap_feature], use_container_width=True)

            plot_name = 'single_shap'
            filename = f'{plot_name}_{selected_shap_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
            single_value_plot = Image.open(f'./web_app/data_upload/temp/{filename}')
            col1.image(single_value_plot)

            # plot_name = 'scatter'
            # filename = f'{plot_name}_{selected_shap_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
            # shap_scatter_plot = Image.open(f'./web_app/data_upload/temp/{filename}')
            # col1.image(shap_scatter_plot)

            # plot_name = 'dependence'
            # filename = f'{plot_name}_{selected_shap_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
            # shap_dependence_plot = Image.open(f'./web_app/data_upload/temp/{filename}')
            # col1.image(single_value_plot)

            # plot_name = 'waterfall'
            # filename = f'{plot_name}_{selected_shap_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
            # waterfall_plot = Image.open(f'./web_app/data_upload/temp/{filename}')
            # col2.image(waterfall_plot)

            # plot_name = 'beeswarm'
            # filename = f'{plot_name}_{selected_shap_feature}_{classification_method}_{cohort_title}_{sampling_title}.png'
            # beeswarm_plot = Image.open(f'./web_app/data_upload/temp/{filename}')
            # col2.image(beeswarm_plot)
        else:
            st.write('Shapley calculation can take up to 1 minute.')

        st.markdown('___')
