import streamlit as st
import datetime
import pandas as pd

from PIL import Image


def write_warning():
    color1 = '#E75919'
    color2 = '#EE895C'
    color3 = '#FFFFFF'
    text = 'Before starting the analysis, we strongly recommend to load the desired dataset in advance. You can do this in the "Data Loader" tab.'
    st.markdown(
        f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:22px;border-radius:2%;">{text}</p>',
        unsafe_allow_html=True)


def display_table(selected_set_name: str, selected_column):
    info_p3 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    selected_column.markdown(info_p3)
    # temp_ca, cache_file_name = CompleteAnalysis.get_analysis(selected_label="fake_label", selected_tool='fake_tool', selected_set=selected_set_name)

    general_info = {'Hospital System': ['Number of patients', 'Number of septic patients', 'Sepsis prevalence in %',
                                        'Number of entries', 'Number of NaNs', 'NaN prevalence in %',
                                        'Total hours recorded', 'Average hospital stay duration (hours)']
                    }

    df_general_info = pd.DataFrame(general_info)

    # Lösung1
    # df_general_info[selected_set_name] = df_general_info[selected_set_name].astype(str)
    # df_general_info[selected_set_name] = df_general_info[selected_set_name].str.replace('.0', ' ', regex=False)

    # df_general_info = df_general_info.style.format(na_rep='MISSING')
    # selected_column.dataframe(df_general_info)

def start_loading(selected_set_list, selected_label_list, selected_column: str = None):
    total_start_time = datetime.datetime.now()
    print("Loading started at time:", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
    for unique_set in selected_set_list:
        if '0_Load all labels (long waiting time!)' in selected_label_list:
            pass

        for label in selected_label_list:
            start_time = datetime.datetime.now()
            # ca.get_analysis(selected_label=label, selected_set=unique_set, selected_tool='none')
            difference_time = datetime.datetime.now() - start_time
            print("Loading of", unique_set, label, "took: ", str(difference_time).replace(" ", "_").replace(":", "-"))
    print("\nLoading finished at time:", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
    total_difference_time = datetime.datetime.now() - total_start_time
    print("Complete loading took: ", str(total_difference_time).replace(" ", "_").replace(":", "-"))



def home_page():
    col1, col2 = st.columns((0.71, 0.29))

    # Column 1: General Info
    col1.markdown("<h2 style='text-align: left; color: black;'>Master Thesis Dashboard</h2>", unsafe_allow_html=True)
    col1.markdown("This dashboard visualizes the results of the Master Thesis "
                  "**Analysis of Machine Learning Prediction Quality for Automated Subgroups within the MIMIC III Dataset**. "
                  "It was published by Jakob Vanek at Goethe University, 2023. "
                  "The guiding supervisor was Prof. Dr. Lena Wiese, Professorship for Database Technologies and Data Analytics.")
    col1.markdown("For further information, please contact via https://jayveezy1.github.io/")

    # Column 2: Goethe Logo
    feature_graphic = Image.open(r'./web_app/web_supplement/logo.jpg')
    col2.image(feature_graphic, width=350)  # , caption='Frankfurt am Main, 2023'

    # Abstract
    st.markdown("<h2 style='text-align: left; color: black;'>Abstract</h2>", unsafe_allow_html=True)
    st.markdown('The motivation for this master thesis is to explore the potential of predictive data analytics in the field of medicine. '
                'Recent advancements in computer science, such as electronic health record systems, have facilitated the development of '
                'novel tools that can significantly enhance the efficacy of healthcare professionals in their daily work. '
                'The possible boundaries of this field remain largely undefined, which presents a promising and compelling field for research. '
                'The MIMIC III dataset offers a sufficiently extensive foundation for the construction of prediction models, such as Random Forest, '
                'XGBOOST and deep learning networks. These models shall be evaluated based on their effectiveness, as well as their fairness. '
                'Finally, automated subgroup clustering will be conducted, followed by a comparative analysis of the prediction performance '
                'for each of the subgroups. ')

    # MIMIC III Dataset
    st.markdown("<h2 style='text-align: left; color: black;'>MIMIC III Dataset</h2>", unsafe_allow_html=True)
    st.markdown('This master thesis analysis is founded on data from the Medical Information Mart for Intensive Care (MIMIC-III) dataset, '
                'version 1.4: https://physionet.org/content/mimiciii/1.4/. The data was collected in the Beth Israel Deaconess '
                'Medical Center, Boston, between 2001 and 2012. It comprises over 58.000 hospital admissions of over 45.000 individual, '
                'deidentified patients who stayed in critical care units. A multitude of features was measured per patient on an hourly basis.'
                'The selected use case for the master thesis is mortality prediction of general stroke, combining hemorrhagic and ischemic cases. '
                'Essential preprocessing steps and filtering were conducted in the backend which led to approximately 2400 individual icu-stays. '
                'Average data is used for the analysis, but timeseries data is available for further research.')

    # How to use this Dashboard
    st.markdown("<h2 style='text-align: left; color: black;'>How to use the Dashboard</h2>", unsafe_allow_html=True)
    st.markdown('This dashboard can be used to interactively recreate the results of the thesis. '
                'Within the section "Data Loader" a user can choose different underlying filtering steps. '
                'Once the analysis is conducted, the results can be found in the respective subpages. The graphics are cached, which greatly enhances display time. ')

    ### End of page ###
