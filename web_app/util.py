import os
import re
import uuid

import pandas as pd
import streamlit as st

from objects.patients import Patient
from step_2_preprocessing.preprocessing_functions import get_preprocessed_avg_cohort


def start_streamlit_frontend(use_this_function: False):
    if use_this_function:
        print(f'\nSTATUS: Starting Frontend: ')
        os.system('streamlit run web_app/app.py')


def get_default_values(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable):       # selected_variable not used but better kept here
    default_values = [x for x in ALL_FEATURES if x not in ALL_DEPENDENT_VARIABLES]

    # remove these for standard selection, keep: ethnicity, gender, oasis, gcs, o2, heart rate, anion gap, sodium, white blood cells
    try:
        default_values.remove('age')
    except ValueError as e:
        pass
    try:
        default_values.remove('stroke_type')
    except ValueError as e:
        pass
    try:
        default_values.remove('admission_type')
    except ValueError as e:
        pass
    try:
        default_values.remove('dbsource')
    except ValueError as e:
        pass
    try:
        default_values.remove('electivesurgery')
    except ValueError as e:
        pass
    try:
        default_values.remove('mechvent')
    except ValueError as e:
        pass
    try:
        default_values.remove('Bicarbonate')
    except ValueError as e:
        pass
    try:
        default_values.remove('Chloride (whole blood)')
    except ValueError as e:
        pass
    try:
        default_values.remove('Creatinine')
    except ValueError as e:
        pass
    try:
        default_values.remove('icustay_id')
    except ValueError as e:
        pass
    try:
        default_values.remove('patientweight')      # no statistical significance
    except ValueError as e:
        pass
    try:
        default_values.remove('hypertension_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('insurance')
    except ValueError as e:
        pass
    try:
        default_values.remove('religion')
    except ValueError as e:
        pass
    try:
        default_values.remove('Respiratory Rate')
    except ValueError as e:
        pass
    try:
        default_values.remove('marital_status')
    except ValueError as e:
        pass
    try:
        default_values.remove('Arterial Blood Pressure mean')
    except ValueError as e:
        pass
    try:
        default_values.remove('cancer_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('drug_abuse_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('sepsis_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('obesity_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('diabetes_flag')
    except ValueError as e:
        pass
    try:
        default_values.remove('gauges_total')
    except ValueError as e:
        pass
    try:
        default_values.remove('Glucose (whole blood)')
    except ValueError as e:
        pass

    return default_values


def get_preselection_one():
    return ['Anion Gap', 'ethnicity', 'gcs', 'gender', 'Heart Rate', 'O2 saturation pulseoxymetry', 'oasis',
              'Sodium (whole blood)', 'White Blood Cells']


def get_preselection_two():
    return ['Anion Gap', 'ethnicity', 'gcs', 'gender', 'Heart Rate', 'O2 saturation pulseoxymetry', 'oasis',
              'Sodium (whole blood)', 'White Blood Cells',
              'admission_type', 'age', 'Bicarbonate', 'Chloride (whole blood)', 'Creatinine',
              'electivesurgery', 'mechvent', 'stroke_type']


def insert_feature_selectors(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable):
    st.markdown('___')
    col1, col2, col3 = st.columns((0.25, 0.25, 0.5))

    # parameter on_change would be useful to set checker_2=False and reset previous default_values
    checker_1 = col1.checkbox(label='Feature Pre-Selection 1', value=True)   # , help='Select the optimized feature selection from the thesis.')
    checker_2 = col2.checkbox(label='Feature Pre-Selection 2', value=False)     # , help='Select the original feature selection from the thesis.')
    if checker_1 and not checker_2:
        # selected_features = st.multiselect(label='Select features', options=ALL_FEATURES, default=get_preselection_one())
        selected_features = get_preselection_one()
        st.markdown(f'Selected Features: {selected_features}')
    elif checker_2 and not checker_1:
        # selected_features = st.multiselect(label='Select features', options=ALL_FEATURES, default=get_preselection_two())
        selected_features = get_preselection_two()
        st.markdown(f'Selected Features: {selected_features}')
    else:
        # default_values = get_default_values(ALL_FEATURES, ALL_DEPENDENT_VARIABLES, selected_variable)
        default_values = get_preselection_one()
        selected_features = st.multiselect(label='Manually select features', options=ALL_FEATURES, default=default_values)

    return selected_features


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


def add_download_button(position, dataframe, title, cohort_title, keep_index: False):
    try:
        csv_table = dataframe.to_csv(index=keep_index).encode('utf-8')
    except AttributeError:      # if previous function returns None instead of table, no download button possible
        return None

    if position is None:
        col1, col2 = st.columns((0.9, 0.11))
        col2.download_button(label="Download the table", data=csv_table,
                             file_name=f'{cohort_title}_{title}.csv', mime="text/csv") # , key='download-csv')
    else:
        position.download_button(label="Download the table", data=csv_table,
                             file_name=f'{cohort_title}_{title}.csv', mime="text/csv")  # , key='download-csv')

def add_single_feature_filter(selected_cohort, selected_features):
    # st.write(selected_features)
    all_features = selected_cohort.columns.to_list()
    selected_features_for_fairness = st.multiselect(label='Select features',
                                                    options=all_features,
                                                    # default=[],
                                                    max_selections=3)
    # st.write(selected_features_for_fairness)
    for feature in selected_features_for_fairness:
        if feature not in selected_features:
            st.warning(f'Feature {feature} must also be selected at top.')
    if len(selected_features_for_fairness) == 3:
        st.write('Maximum selection of features reached.')

    # Factorize categorical features
    factorization_df = pd.read_excel(
        './supplements/FACTORIZATION_TABLE.xlsx')  # columns: feature	unfactorized_value	factorized_value
    features_to_factorize = pd.unique(factorization_df['feature']).tolist()

    selected_privileged_values = []
    for feature in selected_features_for_fairness:
        available_values = selected_cohort[feature].unique()
        if feature in features_to_factorize:
            factorized_values = factorization_df.loc[factorization_df['feature'] == feature][
                'factorized_value'].to_list()  # might be helpful to display these in the label
            available_values = get_unfactorized_values(feature, factorization_df)

        protected_values_for_feature = st.multiselect(label=f'Select protected values for {feature}',
                                                      options=available_values)
        selected_privileged_values.append(protected_values_for_feature)
        if len(protected_values_for_feature) < 1:
            st.warning('Choose one value/class for each selected features.')
        elif len(protected_values_for_feature) > 1:
            st.warning('Warning: For most categorical features only a selection of one attribute is sensible.')

    selected_filter_features = selected_features_for_fairness
    selected_filter_values = selected_privileged_values

    return selected_filter_features, selected_filter_values

def get_unfactorized_values(feature, factorization_df):
    unfactorized_values = []
    temp_unfactorized_df = factorization_df.loc[factorization_df['feature'] == feature]
    for factorized_value in temp_unfactorized_df['factorized_value'].to_list():
        temp_unfact_value = temp_unfactorized_df.loc[temp_unfactorized_df['factorized_value'] == factorized_value, 'unfactorized_value'].item()
        unfactorized_values.append(temp_unfact_value)

    return unfactorized_values
