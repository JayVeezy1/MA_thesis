import streamlit as st
from PIL import Image

from web_app.subpages.Subgroup_Analysis import subgroup_analysis_page
from web_app.subpages.Classification import classification_page
from web_app.subpages.Clustering import clustering_page
from web_app.subpages.Data_Analysis import data_analysis_page
from web_app.subpages.Data_Upload import data_upload_page
from web_app.subpages.Fairness_Analysis import fairness_page
from web_app.subpages.Home import home_page
from web_app.util import create_st_button


def create_streamlit_frontend():
    # General Config
    st.set_page_config(page_title='Master Thesis Vanek',
                       page_icon=Image.open('web_app/web_supplement/favicon.png'),
                       layout='wide',
                       initial_sidebar_state='expanded',
                       menu_items = {'About': "This Dashboard was developed to display the results of the master thesis "
                                              "'Analysis of Machine Learning Prediction Quality for Automated Subgroups within the MIMIC III Dataset' "
                                              "by Jakob Vanek, 2023. For further information, please contact via https://jayveezy1.github.io/"})

    # Sidebar Page Checkboxes
    st.sidebar.title('Main Menu')
    pages = [{'title': 'Home', 'function': home_page},
             {'title': 'Data Upload', 'function': data_upload_page},
             {'title': 'General Data Analysis', 'function': data_analysis_page},
             {'title': 'Clustering', 'function': clustering_page},
             {'title': 'Classification', 'function': classification_page},
             {'title': 'Fairness Analysis', 'function': fairness_page},
             {'title': 'Subgroup Analysis', 'function': subgroup_analysis_page}]
    menu = st.sidebar.radio(label='Select Page', options=pages, format_func=lambda page: page['title'], label_visibility='collapsed')
    menu['function']()

    # Add Spaces Before Links
    for i in range(0, 25):
        st.sidebar.markdown(' ')
    st.sidebar.markdown('---')

    # Add Links to Sidebar
    st.sidebar.markdown("## Further Information")
    information_link_dict = {
            'Master Thesis Paper': "https://www.overleaf.com/project/637645c65a754832b1e27443",
            'GitHub Repository': "https://github.com/JayVeezy1/MA_thesis",
            'Professorship DBDA': "http://www.dbda.cs.uni-frankfurt.de/index.html"}
    for link_text, link_url in information_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)
    # st.sidebar.markdown('---')

create_streamlit_frontend()
