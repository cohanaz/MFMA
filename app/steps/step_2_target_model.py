# app/steps/step_2_target_model.py

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from app.utils.training import train_target_model

def run():
    if st.session_state.dataset is not None:
        st.write("Preview of the dataset:")
        st.dataframe(st.session_state.dataset.head())

        default_index = 0
        if st.session_state.target_column in st.session_state.dataset.columns:
            default_index = st.session_state.dataset.columns.get_loc(st.session_state.target_column)

        numeric_columns = st.session_state.dataset.select_dtypes(include=['number']).columns
        if st.session_state.target_column in numeric_columns:
            default_numeric_index = list(numeric_columns).index(st.session_state.target_column)
        else:
            default_numeric_index = 0
        target_column = st.selectbox("Select target column (numerical only):", numeric_columns, index=default_numeric_index)

        st.write("Select target model type:")
        # Determine default index based on session state
        model_options = ["XGBoost", "Random Forest"]
        if "target_model_type" in st.session_state and st.session_state.target_model_type in model_options:
            default_index = model_options.index(st.session_state.target_model_type)
        else:
            default_index = 0
        model_type = option_menu(
            menu_title="",
            options=model_options,
            icons=["bar-chart-line", "tree"],
            menu_icon="cast",
            default_index=default_index,
            orientation="horizontal",
            styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6", "justify-content": "center"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "5px",
                "padding": "10px",
                "--hover-color": "#E8F5E9",
            },
            "nav-link-selected": {
                "background-color": "#4CAF50",
                "color": "white",
                "font-weight": "bold",
                "border-radius": "6px",
            },
            }
        )
  

        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            if st.button("Train Target Model", use_container_width=True):
                if model_type == "XGBoost":
                    model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_target_model(st.session_state.dataset, target_column, model_type="xgb")
                else:
                    model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_target_model(st.session_state.dataset, target_column, model_type="rf")

                st.session_state.target_model = model
                st.session_state.target_X_train = X_train
                st.session_state.target_X_test = X_test
                st.session_state.target_X_ext = X_ext
                st.session_state.target_y_train = y_train
                st.session_state.target_y_test = y_test
                st.session_state.target_y_ext = y_ext
                st.session_state.target_r2 = r2_score
                st.session_state.target_of_ratio = of_ratio
                st.session_state.target_model_trained = True

                st.success(f"Target model trained successfully! R² score: {r2_score:.2f}, Overfit ratio: {of_ratio:.0f}")
                st.session_state.target_model_type = model_type
                st.session_state.target_column = target_column

        # Only show Next button if model is trained
        next_enabled = st.session_state.get("target_model_trained", False)
    else:
        st.info("Please load a dataset in Step 1 first.")
        next_enabled = False

    # Navigation buttons after Step 2
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True) and st.session_state.active_step > 0:
            st.session_state.active_step -= 1
            st.rerun()

    with col2:
        next_enabled = "target_model" in st.session_state
        if st.button("Next ➡", use_container_width=True, disabled=not next_enabled) and st.session_state.active_step < 4:
            st.session_state.active_step += 1
            st.rerun()

