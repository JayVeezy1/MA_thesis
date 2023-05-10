import streamlit as st

from PIL import Image


def home_page():
    col1, col2 = st.columns((0.71, 0.29))

    # Column 1: General Info
    col1.markdown("<h1 style='text-align: left; color: black;'>Master Thesis Dashboard</h1>", unsafe_allow_html=True)
    col1.markdown("This dashboard visualizes the results of the Master Thesis "
                  "**Analysis of Machine Learning Prediction Quality for Automated Subgroups within the MIMIC III Dataset**. "
                  "It was developed by Jakob Vanek at Goethe University, 2023. "
                  "The guiding supervisor was Prof. Dr. Lena Wiese, Professorship for Database Technologies and Data Analytics.")
    col1.markdown("For further information, please contact via https://jayveezy1.github.io/")

    # Column 2: Goethe Logo
    feature_graphic = Image.open(r'./web_app/web_supplement/logo.jpg')
    col2.image(feature_graphic, width=350)  # , caption='Frankfurt am Main, 2023'
    # st.markdown('___')

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
    # st.markdown('___')

    # MIMIC III Dataset
    st.markdown("<h2 style='text-align: left; color: black;'>MIMIC III Dataset</h2>", unsafe_allow_html=True)
    st.markdown('This master thesis analysis is founded on data from the Medical Information Mart for Intensive Care (MIMIC-III) dataset, '
                'version 1.4: https://physionet.org/content/mimiciii/1.4/. The data was collected in the Beth Israel Deaconess '
                'Medical Center, Boston, between 2001 and 2012. It comprises over 58.000 hospital admissions of over 45.000 individual, '
                'deidentified patients who stayed in critical care units. A multitude of features was measured per patient on an hourly basis.'
                'The selected use case for the master thesis is mortality prediction of general stroke, combining hemorrhagic and ischemic cases. '
                'Essential preprocessing steps and filtering were conducted in the backend which led to approximately 2600 individual icu-stays. '
                'Average data is used for the analysis, but timeseries data is available for further research.')
    # st.markdown('___')

    # How to use this Dashboard
    st.markdown("<h2 style='text-align: left; color: black;'>How to use the Dashboard</h2>", unsafe_allow_html=True)
    st.markdown('This dashboard can be used to interactively recreate the results of the thesis. '
                'Within the section "Data Upload" a user has to upload a "avg_patient_cohort.csv" file. '
                'This cohort can be created with the python function "get_avg_patient_cohort()" once the Mimic III dataset is loaded into python. '
                'The results of the analysis can be found in the respective subpages. '
                'Depending on the use case, the calculation of the plots can can take up to 2 minutes. '
                'Afterwards these calculations are cached, which greatly enhances display time. ')

    st.markdown('___')
    ### End of page ###
