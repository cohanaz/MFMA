# app/steps/step_2_target_model.py

import numpy as np
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from dataset_app.utils.training import train_target_model, generate_target_models, train_XGB_model, train_RF_model

def run():
    if st.session_state.dataset is not None:
        st.write("Preview of the dataset:")
        st.dataframe(st.session_state.dataset.head())

        numeric_columns = st.session_state.dataset.select_dtypes(include=['number']).columns
        default_numeric_index = (
            list(numeric_columns).index(st.session_state.target_column)
            if st.session_state.target_column in numeric_columns
            else 0
        )
        target_column = st.selectbox("Select target column (numerical only):", numeric_columns, index=default_numeric_index)

        model_options = ["XGBoost", "Random Forest"]
        default_model_index = model_options.index(st.session_state.get("dataset_model_type", "XGBoost"))
        model_type = option_menu(
            menu_title="",
            options=model_options,
            icons=["bar-chart-line", "tree"],
            menu_icon="cast",
            default_index=default_model_index,
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

        # ⚙️ פרמטרים לבחירה
        col1, _, col2, _, col3 = st.columns([2, 1, 2, 1, 2])
        with col1:
            n_target_models = st.slider("# models to generate:", min_value=10, max_value=100, value=10, step=10)
        with col2:
            owned_data_ratio = st.slider(f"% owned data:", 0.2, 0.8, 0.5, 0.1)
        with col3:
            holdout_data_ratio = st.slider(f"% holdout data:", 0.0, 0.4, 0.0, 0.1)

        owned_to_external_ratio = [1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0]

        # Generate random hyperparameters for each shadow model
        np.random.seed(0)
        target_params = []
        for i in range(n_target_models):
            params = {
                "Seed": 42 + i,
                "Max Depth": np.random.randint(3, 8),
                "Estimators": int(np.random.choice([50, 100, 150, 200, 250, 300, 350, 400])),
                "Owned to External Ratio": owned_to_external_ratio[i % len(owned_to_external_ratio)]
            }
            target_params.append(params)

        st.markdown("---")
        st.markdown("### Target Models Hyperparameters")
        params_df = pd.DataFrame(target_params)
        params_df.index = [f"{i+1}" for i in range(len(params_df))]
        params_df.index.name = "#Model"

        if model_type == "XGBoost":
            train_func = train_XGB_model
        else:
            train_func = train_RF_model

        centered_col = st.columns([1, 4, 1])[1]
        with centered_col:
            st.dataframe(params_df, height=160)
            if st.button("Generate Target Models", use_container_width=True):
                models, data_splits, model_stats, owned, external = generate_target_models(
                    train_function=train_func,
                    dataset=st.session_state.dataset,
                    target_col=target_column,
                    owned_data_ratio=owned_data_ratio,
                    holdout_ratio=holdout_data_ratio,
                    param_dicts=target_params,
                    random_state=42
                )

                # שמירה ב-session_state
                st.session_state.target_models = models
                st.session_state.target_splits = data_splits
                st.session_state.target_stats = model_stats
                st.session_state.dataset_owned = owned
                st.session_state.dataset_external = external
                st.session_state.target_model_type = model_type
                st.session_state.target_column = target_column
                st.session_state.target_models_trained = True

                st.success(f"{len(models)} models generated successfully.")

        # Only show Next button if model is trained
        next_enabled = st.session_state.get("target_models_trained", False)
        
    else:
        st.info("Please load a dataset in Step 1 first.")
        next_enabled = False
        
    st.markdown("---")
    # Navigation buttons after Step 2
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True) and st.session_state.active_step > 0:
            st.session_state.target_model_type = model_type
            st.session_state.active_step -= 1
            st.rerun()

    with col2:
        next_enabled = "target_models_trained" in st.session_state
        if st.button("Next ➡", use_container_width=True, disabled=not next_enabled) and st.session_state.active_step < 4:
            st.session_state.active_step += 1
            st.rerun()

