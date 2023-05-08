import os

import pandas as pd
import streamlit as st
from web_app.util import get_avg_cohort_cache
from step_4_classification.classification import get_classification_report, get_confusion_matrix
from step_4_classification.classification_deeplearning import get_classification_report_deeplearning, \
    get_DL_confusion_matrix
from step_3_data_analysis.clustering import calculate_cluster_kmeans, preprocess_for_clustering, \
    calculate_cluster_kprot, calculate_cluster_dbscan
from step_3_data_analysis.data_visualization import calculate_pacmap
from step_5_fairness.fairness_analysis import get_fairness_report

def data_upload_page():
    ## Upload Dataset
    st.markdown("<h2 style='text-align: left; color: black;'>Data Upload</h2>", unsafe_allow_html=True)
    st.markdown("Please upload Your dataset here. It should be a .csv file of the average patient cohort, where each row is an individual patient with an icustay_id as key. "
                "The dataset is saved in the data_upload folder in the project files. There can always only be one dataset in usage. "
                "Important: Be careful not to commit the uploaded avg_patient_cohort.csv into a public repository or share it in any way, as the MIMIC-III data is protected. "
                "In addition, You can use the ./supplements/features_preprocessing.xlsx table in the project folder to define selected features and the "
                "./supplements/factorization_table.xlsx to determine the factorization values.")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        file_df = pd.read_csv(uploaded_file)
        st.markdown("The following file was uploaded to ./web_app/data_upload/exports/frontend/avg_patient_cohort.csv")
        st.write(file_df)
        upload_filename_string: str = './web_app/data_upload/exports/frontend/avg_patient_cohort.csv'

        upload_filename = upload_filename_string.encode()
        with open(upload_filename, 'w', newline='') as output_file:
            file_df.to_csv(output_file, index=False)

    ## Clear Cache
    st.markdown("<h2 style='text-align: left; color: black;'>Clear Cache</h2>", unsafe_allow_html=True)
    st.markdown("Streamlit cacheing enables extremely fast display of tables and plots. Be certain to clear the cache when using a new dataset.")

    if st.button(label='Clear Cache'):
        # This clears all previous cached data
        get_avg_cohort_cache.clear()
        # Classification
        get_confusion_matrix.clear()
        get_classification_report.clear()
        get_classification_report_deeplearning.clear()
        get_DL_confusion_matrix.clear()
        get_fairness_report.clear()

        # Clustering
        calculate_cluster_kmeans.clear()
        calculate_cluster_kprot.clear()
        calculate_cluster_dbscan.clear()
        preprocess_for_clustering.clear()
        calculate_pacmap.clear()


    ## Delete Dataset
    st.markdown("<h2 style='text-align: left; color: black;'>Delete Patient Cohort File</h2>", unsafe_allow_html=True)
    st.markdown('You can delete the previously uploaded avg_patient_cohort.csv file here.')
    if st.button(label='Delete Uploaded File'):
        file_path = './web_app/data_upload/exports/frontend/avg_patient_cohort.csv'
        if os.path.isfile(file_path):
            os.remove(file_path)
            st.markdown('File was successfully deleted.')
        else:
            st.markdown(f'No file exists in {file_path}')
