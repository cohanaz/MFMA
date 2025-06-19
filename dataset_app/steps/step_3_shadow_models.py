# step_3_shadow_models.py

import streamlit as st
import numpy as np
import pandas as pd
from dataset_app.utils.training import train_RF_model, train_XGB_model, train_shadow_models

def run():
    st.subheader("Step 3: Train Shadow Models")

    st.session_state.num_shadow_models = st.slider("Number of shadow models:", min_value=1, max_value=10, value=3)
    
    seeds = [42 + i for i in range(st.session_state.num_shadow_models)]

    # Generate random hyperparameters for each shadow model
    np.random.seed(0)
    used_depths = set()
    used_estimators = set()
    shadow_params = []
    for seed in seeds:
        while True:
            max_depth = np.random.randint(3, 10)
            if max_depth not in used_depths:
                used_depths.add(max_depth)
                break

        while True:
            estimators = int(np.random.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]))
            if estimators not in used_estimators:
                used_estimators.add(estimators)
                break

        params = {
            "Seed": seed,
            "Max Depth": max_depth,
            "Estimators": estimators
        }
        shadow_params.append(params)

    st.markdown("---")
    st.markdown("### Shadow Model Hyperparameters")
    params_df = pd.DataFrame(shadow_params)
    params_df.index = [f"Shadow Model {i+1}" for i in range(len(params_df))]
    st.dataframe(params_df)

    # Track if shadow models are trained
    if "shadow_models_trained" not in st.session_state:
        st.session_state.shadow_models_trained = False

    centered_col = st.columns([1, 3, 1])[1]
    with centered_col:
        if st.button("Train Shadow Models", use_container_width=True):
            param_dicts = [
                {
                    "rand_stat": row["Seed"],
                    "max_d": row["Max Depth"],
                    "n_est": row["Estimators"]
                }
                for _, row in params_df.iterrows()
            ]

            if st.session_state.target_model_type == "XGBoost":
                train_func = train_XGB_model
            else:
                train_func = train_RF_model

            models, splits, stats = train_shadow_models(
                train_function=train_func,
                data=st.session_state.dataset,
                target=st.session_state.target_column,
                param_dicts=param_dicts,
                ext_size=0.2
            )

            st.session_state.shadow_models = models
            st.session_state.shadow_splits = splits
            st.session_state.shadow_stats = stats
            st.session_state.shadow_models_trained = True

            # st.markdown("### Shadow Model Performance")
            # perf_df = pd.DataFrame(stats, columns=["Overfit Ratio", "R² Score"], index=[f"Shadow Model {i+1}" for i in range(len(stats))])
            # # Add target model row at the top
            # target_row = pd.DataFrame({
            #     "Overfit Ratio": [st.session_state.target_of_ratio],
            #     "R² Score": [st.session_state.target_r2]
            # }, index=["**Target Model**"])

            # perf_df = pd.concat([target_row, perf_df])
            # st.table(perf_df.style.format({"Overfit Ratio": "{:.0f}", "R² Score": "{:.2f}"}))

            st.success("Shadow models trained and stored successfully!")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_step -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡", use_container_width=True, disabled=not st.session_state.shadow_models_trained):
            st.session_state.active_step += 1
            st.rerun()