# app/main.py

import streamlit as st
import importlib
from dataset_app.utils.general import display_resources_usage

def run_app():

    # Define the steps and corresponding module paths
    steps = {
        "Choose Dataset": "step_1_dataset",
        "Train Target Models": "step_2_target_model",
        "Train Shadow Models": "step_3_shadow_models",
        "Extract Features": "step_4_feature_extraction",
        "Train Inference Model": "step_5_inference_model",
        "Membership Inference": "step_6_run_inference"
    }

    # Determine completion status of each step
    completed = {
        "Choose Dataset": "dataset" in st.session_state,
        "Train Target Models": "target_model" in st.session_state,
        "Train Shadow Models": "shadow_models" in st.session_state,
        "Extract Features": "attack_features" in st.session_state,
        "Train Inference Model": "inference_model_trained" in st.session_state,
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

    joined_desc = "<br>".join(desc_parts)

    short_names = {
        "Prediction": "PRED",
        "error": "ERR",
        "missing": "MISS",
        "Ens_var": "ENS",
        "AUG_preds": "AUG_preds",
        "AUG_stats": "AUG_stats"
    }

    if "selected_feature_groups" in st.session_state:
        selected_short = [short_names.get(g, g[:4].upper()) for g in st.session_state.selected_feature_groups]
        inference_desc = " | ".join(selected_short)
    else:
        inference_desc = ""

    if "target_models" in st.session_state and "owned_model_ratio" in st.session_state and "owned_ratio" in st.session_state:
        n_models = len(st.session_state.target_models)
        n_owned_models = int(n_models * st.session_state.owned_model_ratio)
        owned_ratio_pct = int(st.session_state.owned_ratio * 100)
        model_type = st.session_state.get("target_model_type", "")
        train_target_desc = f"{model_type}<br>{owned_ratio_pct}% data owned<br>{n_owned_models}/{n_models} models owned"
    else:
        train_target_desc = st.session_state.get("target_model_type", "")

    
    # Define optional descriptions for each step
    step_descriptions = {
        "Choose Dataset": f'{st.session_state.get("dataset_name", "")} <br> {len(st.session_state.dataset)} rows' 
                            if "dataset" in st.session_state else "",
        "Train Target Models": train_target_desc,
        "Train Shadow Models": f'{len(st.session_state.shadow_models)} models' if "shadow_models" in st.session_state else "",
        "Extract Features": f'{joined_desc}' if desc_parts else "",
        "Train Inference Model": inference_desc,
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
                width: 100px; height: auto; 
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


