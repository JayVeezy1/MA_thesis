import os
import re
import uuid
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
def get_avg_cohort_cache(project_path, use_case_name, features_df, delete_existing_cache, selected_patients=[]):

    # Directly get avg_cohort file
    # TODO: has to be uploaded first in "Data Loader" by user
    raw_avg_cohort = Patient.get_avg_patient_cohort(project_path=project_path,
                                                         use_case_name=use_case_name,
                                                         features_df=features_df,
                                                         delete_existing_cache=delete_existing_cache,
                                                         selected_patients=selected_patients)    # empty=all
    # Scaling (maybe make dependent on user?)
    scaled_avg_cohort = Patient.get_avg_scaled_data(avg_patient_cohort=raw_avg_cohort,
                                                    features_df=features_df)

    # Preprocessing (maybe make dependent on user?)
    preprocessed_avg_cohort = get_preprocessed_avg_cohort(avg_cohort=scaled_avg_cohort,
                                                          features_df=features_df)

    # TODO: add filtering depending on selectors in web_app
    filtered_avg_cohort = preprocessed_avg_cohort

    return filtered_avg_cohort
