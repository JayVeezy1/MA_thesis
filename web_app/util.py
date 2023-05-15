import os
import re
import uuid

import pandas as pd
import streamlit as st

from objects.patients import Patient
from step_1_setup_data import cache_IO
from step_1_setup_data.cache_IO import load_data_from_cache
from step_2_preprocessing.preprocessing_functions import get_preprocessed_avg_cohort


def start_streamlit_frontend(use_this_function: False):
    if use_this_function:
        print(f'\nSTATUS: Starting Frontend: ')
        os.system('streamlit run web_app/app.py')


# From https://github.com/JayVeezy1/rascore
def create_st_button(link_text, link_url, hover_color="#e78ac3", st_col=None):

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    button_css = f"""
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }}
            #{button_id}:hover {{
                border-color: {hover_color};
                color: {hover_color};
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: {hover_color};
                color: white;
                }}
        </style> """
    html_str = f'<a href="{link_url}" target="_blank" id="{button_id}";>{link_text}</a><br></br>'
    if st_col is None:
        st.markdown(button_css + html_str, unsafe_allow_html=True)
    else:
        st_col.markdown(button_css + html_str, unsafe_allow_html=True)


@st.cache_data
def get_avg_cohort_cache(project_path, use_case_name, features_df, selected_database, selected_stroke_type,
                         delete_existing_cache, selected_patients=None):
    if selected_patients is None:
        selected_patients = []

    # Directly get avg_cohort file (uploaded by user in "Data Loader")
    raw_avg_cohort = Patient.get_avg_patient_cohort(project_path=project_path,
                                                         use_case_name=use_case_name,
                                                         features_df=features_df,
                                                         delete_existing_cache=delete_existing_cache,
                                                         selected_patients=selected_patients)    # empty=all
    # Scaling
    scaled_avg_cohort = Patient.get_avg_scaled_data(avg_patient_cohort=raw_avg_cohort,
                                                    features_df=features_df)

    # Preprocessing
    filtered_avg_cohort = get_preprocessed_avg_cohort(avg_cohort=scaled_avg_cohort,
                                                      features_df=features_df,
                                                      selected_database=selected_database,
                                                      selected_stroke_type=selected_stroke_type)


    return filtered_avg_cohort


def add_download_button(position, dataframe, title, cohort_title):
    try:
        csv_table = dataframe.to_csv(index=False).encode('utf-8')
    except AttributeError:      # if previous function returns None instead of table, no download button possible
        return None

    if position is None:
        col1, col2 = st.columns((0.9, 0.11))
        col2.download_button(label="Download the table", data=csv_table,
                             file_name=f'{cohort_title}_{title}.csv', mime="text/csv") # , key='download-csv')
    else:
        position.download_button(label="Download the table", data=csv_table,
                             file_name=f'{cohort_title}_{title}.csv', mime="text/csv")  # , key='download-csv')


def get_unfactorized_values(feature, factorization_df):
    unfactorized_values = []
    temp_unfactorized_df = factorization_df.loc[factorization_df['feature'] == feature]
    for factorized_value in temp_unfactorized_df['factorized_value'].to_list():
        temp_unfact_value = temp_unfactorized_df.loc[temp_unfactorized_df['factorized_value'] == factorized_value, 'unfactorized_value'].item()
        unfactorized_values.append(temp_unfact_value)

    return unfactorized_values
