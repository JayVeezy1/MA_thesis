import streamlit as st

def data_loader_page():
    st.markdown("<h2 style='text-align: left; color: black;'>Data Loader</h2>", unsafe_allow_html=True)
    st.markdown("This is the Data Loader page.")


    # start with get_avg_cohort_cache.clear() to remove previous caches
