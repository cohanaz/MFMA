# app/main.py

import streamlit as st
import importlib

def run_app():
    # Define the steps and corresponding module paths
    steps = {
        "Choose Dataset": "step_1_dataset",
        "Train Target Model": "step_2_target_model",
        "Train Shadow Models": "step_3_shadow_models",
        "Extract Features": "step_4_feature_extraction",
        "Membership Inference": "step_5_mia_attack"
    }

    # Determine completion status of each step
    completed = {
        "Choose Dataset": "dataset" in st.session_state,
        "Train Target Model": "target_model" in st.session_state,
        "Train Shadow Models": "shadow_models" in st.session_state,
        "Extract Features": "attack_features" in st.session_state,
        "Membership Inference": "inference_results" in st.session_state
    }

    desc_parts = []

    # אם יש אוגמנטציות
    aug_count = st.session_state.get("aug_count")
    if aug_count is not None:
        desc_parts.append(f"AUG:{aug_count}")

    # אם יש רמות רעש
    noise_levels = st.session_state.get("noise_levels")
    if noise_levels is not None:
        desc_parts.append(f"σ: {'/'.join([f'{sigma}' for sigma in noise_levels])}")

    # אם יש missing info
    n_missing = st.session_state.get("n_missing")
    missing_strategy = st.session_state.get("missing_strategy")
    if n_missing is not None and missing_strategy is not None:
        desc_parts.append(f"MISS: {n_missing}-{missing_strategy}")

    joined_desc = " | ".join(desc_parts)

    # Define optional descriptions for each step
    step_descriptions = {
        "Choose Dataset": st.session_state.get("dataset_name", ""),
        "Train Target Model": st.session_state.get("target_model_type", ""),
        "Train Shadow Models": f'{len(st.session_state.shadow_models)} models' if "shadow_models" in st.session_state else "",
        "Extract Features": f'{joined_desc}' if desc_parts else "",
        "Membership Inference": ""  # ניתן להוסיף מאוחר יותר
    }

    # Track current active step
    if "active_step" not in st.session_state:
        st.session_state.active_step = 0

    # Build the step bar display
    step_labels = list(enumerate(steps.keys(), start=1))
    step_bar = "<div style='display: flex; gap: 10px; justify-content: center;'>"
    for i, (num, name) in enumerate(step_labels):
        is_active = (i == st.session_state.active_step)
        bg = "#4CAF50" if is_active else "#E8F5E9"
        color = "white" if is_active else "#4CAF50"
        border = "2px solid #4CAF50" if is_active else "2px solid #E8F5E9"
        step_bar += f"""
            <div style='
                width: 110px; height: auto; 
                background: {bg}; color: {color}; 
                border-radius: 8px; 
                display: flex; flex-direction: column; 
                align-items: center; justify-content: flex-start;
                font-weight: bold; font-size: 15px;
                border: {border};
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                line-height: 1.1;
                text-align: center;
                padding: 10px 5px;
            '>
                <div style='font-size: 18px;'>{num}</div>
                <div style='margin-top: 2px;'>{name}</div>
                <div style='margin-top: 4px; font-size: 12px; font-weight: normal;'>{"[" + step_descriptions[name] + "]" if step_descriptions.get(name) else ""}</div>
            </div>
        """

        if i < len(step_labels) - 1:
            step_bar += "<div style='align-self: center; font-size: 22px; color: #4CAF50;'>&#8594;</div>"
    step_bar += "</div>"
    st.markdown(" ")
    st.markdown(step_bar, unsafe_allow_html=True)

    st.markdown("---")

    # Load and run the appropriate step module
    selected_step_key = list(steps.values())[st.session_state.active_step]
    module = importlib.import_module(f"dataset_app.steps.{selected_step_key}")
    module.run()
