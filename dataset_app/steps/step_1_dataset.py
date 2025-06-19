# app/steps/step_1_dataset.py

import streamlit as st
import pandas as pd
from dataset_app.utils.data_loading import (
    load_house_data,
    load_census_data,
    load_abalone_data,
    load_kc_house_data,
    load_diamonds_data
)

def run():
    col1, col2, col3 = st.columns([2, 1, 4])

    with col1:
        dataset_name = st.radio("**Choose a built-in dataset:**", [
            "House", "Census", "Abalone", "KC House", "Diamonds"
        ])

    with col2:
        st.markdown("""
            <div style='display: flex; justify-content: left; align-items: center; height: 100px;'>
                <strong style='font-size: 20px;'>Or</strong>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        uploaded_file = st.file_uploader("**Upload your own CSV file:**", type="csv")

    if dataset_name == "House":
        dataset_name = "House"
        st.session_state.dataset = load_house_data()
        st.session_state.target_column = "SalePrice"
    elif dataset_name == "Census":
        dataset_name = "Census"
        st.session_state.dataset = load_census_data()
        st.session_state.target_column = "Income"
    elif dataset_name == "Abalone":
        dataset_name = "Abalone"
        st.session_state.dataset = load_abalone_data()
        st.session_state.target_column = "Rings"
    elif dataset_name == "KC House":
        dataset_name = "KC House"
        st.session_state.dataset = load_kc_house_data()
        st.session_state.target_column = "price"
    elif dataset_name == "Diamonds":
        dataset_name = "Diamonds"
        st.session_state.dataset = load_diamonds_data()
        st.session_state.target_column = "price"

    if uploaded_file:
        dataset_name = uploaded_file.name[:10]
        st.session_state.dataset = pd.read_csv(uploaded_file)
        st.session_state.target_column = None

    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        # Display dataset stats
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        num_numeric = df.select_dtypes(include=['number']).shape[1]
        num_categorical = df.select_dtypes(include=['object', 'category']).shape[1]

        st.markdown("---")
        st.markdown("**Dataset Summary:**")
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric(label="Records", value=num_rows)
        stat2.metric(label="Total Features", value=num_cols)
        stat3.metric(label="Numerical Features", value=num_numeric)
        stat4.metric(label="Categorical Features", value=num_categorical)

        st.markdown("---")
        st.write("**Dataset Preview:**")
        st.dataframe(st.session_state.dataset.head())

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.dataset_name = dataset_name
            st.session_state.active_step += 1
            st.rerun()

