import pandas as pd
import streamlit as st

from step_1_setup_data.cache_IO import load_data_from_cache
from step_3_data_analysis.general_statistics import calculate_deaths_table, calculate_feature_overview_table
from web_app.util import get_avg_cohort_cache


def data_analysis_page():
    st.markdown("<h2 style='text-align: left; color: black;'>General Data Analysis</h2>", unsafe_allow_html=True)
    st.markdown("This is the General Data Analysis Page.")

    ## Get Cohort from streamlit cache function
    # TODO: make project path dependent on current working directory? Or some other solution to make dynamic?
    PROJECT_PATH = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'
    FEATURES_DF = pd.read_excel('./supplements/FEATURE_PREPROCESSING_TABLE.xlsx')

    cached_cohort = get_avg_cohort_cache(project_path=PROJECT_PATH,
                                           use_case_name='frontend',
                                           features_df=FEATURES_DF,
                                           selected_patients=[],
                                           delete_existing_cache=False)    # empty = all
    SELECTED_FEATURES = list(cached_cohort.columns)
    SELECTED_DEPENDENT_VARIABLE = 'death_in_hosp'           # TODO: make this dependent on user input


    # TODO: put all these 'get_plot' functions into outer function to check for object-cache? Or check for cache in these functions?
    # tables are really fast, no problem, but maybe cache the plots and classifications?

    ## Deaths DF
    deaths_df = calculate_deaths_table(use_this_function=True,
                           use_case_name='frontend',
                           cohort_title='frontend',
                           selected_cohort=cached_cohort,
                           save_to_file=False)
    deaths_df = deaths_df.reset_index(drop=True)
    st.markdown("<h2 style='text-align: left; color: black;'>Death Cases Dataframe</h2>", unsafe_allow_html=True)
    st.dataframe(data=deaths_df.set_index(deaths_df.columns[0]), use_container_width=True)

    ## General Statistics DF
    st.markdown("<h2 style='text-align: left; color: black;'>General Overview Table</h2>", unsafe_allow_html=True)
    overview_table = calculate_feature_overview_table(use_this_function=True,  # True | False
                                                      selected_cohort=cached_cohort,
                                                      features_df=FEATURES_DF,
                                                      selected_features=SELECTED_FEATURES,
                                                      cohort_title='frontend',
                                                      use_case_name='frontend',
                                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                      save_to_file=False)
    # CSS to inject markdown, this removes index column from table
    hide_table_row_index = """ <style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style> """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(data=overview_table)
