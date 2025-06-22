# app/utils/general.py

import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import psutil

def set_global_target_models(model):
    global _global_target_model
    _global_target_model = model

def get_global_target_models():
    return _global_target_model

def set_global_target_splits(splits):
    global _global_target_splits
    _global_target_splits = splits

def get_global_target_split(split_id):
    return _global_target_splits[split_id]

def set_global_shadow_models(models):
    global _global_shadow_models
    _global_shadow_models = models

def get_global_shadow_model(model_id):
    return _global_shadow_models[model_id]

def set_global_shadow_splits(splits):
    global _global_shadow_splits
    _global_shadow_splits = splits

def get_global_shadow_split(split_id):
    return _global_shadow_splits[split_id]

def set_global_feature_means_list(means_list):
    global _global_feature_means_list
    _global_feature_means_list = means_list

def get_global_feature_means(model_id):
    return _global_feature_means_list[model_id]

def set_global_feature_medians_list(medians_list):
    global _global_feature_medians_list
    _global_feature_medians_list = medians_list

def get_global_feature_medians(model_id):
    return _global_feature_medians_list[model_id]

def set_global_feature_importances(importances):
    global _global_feature_importances
    _global_feature_importances = importances

def get_global_feature_importance(model_id):
    return _global_feature_importances.iloc[model_id]

def display_resources_usage(container=None):

    process = psutil.Process()
    mem_info = process.memory_info()
    used_mb = mem_info.rss / (1024 ** 2)

    total_mem = psutil.virtual_memory().total / (1024 ** 2)
    ram_percent = (used_mb / total_mem) * 100

    cpu_percent = psutil.cpu_percent(interval=0.5)

    ram_text = f"💾 RAM Usage: {ram_percent:.0f}% ({used_mb:.1f} MB)"
    cpu_text = f"🧠 CPU Usage: {cpu_percent:.0f}%"

    _, col2, col3, _ = st.columns([1, 2, 2, 1])
    with col2:
        st.markdown(f"<div style='text-align: center;'>{ram_text}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='text-align: center;'>{cpu_text}</div>", unsafe_allow_html=True)


def stqdm(iterable, total=None, desc=None):
    if total is None:
        total = len(iterable)
    
    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(desc + f" ({total} items):")
    with progress_col:
        progress = st.progress(0)
        for i, item in enumerate(iterable):
            yield item
            progress.progress(int((i + 1) / total * 100))

def check_model_type(model):
    """
    Checks if the given model is an XGBoost model, a Random Forest (RF) model, or of another type.

    Parameters:
    - model: The model instance to check.

    Returns:
    - A string indicating the model type.
    """
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        return 'Random Forest Model'
    elif isinstance(model, xgb.XGBModel):
        return 'XGBoost Model'
    else:
        return 'Other Model Type'

def align_dicts_by_keys(dict1, dict2):
    """
    Verify both dicts have identical keys and return them ordered by dict1's key order.
    
    Returns:
        aligned_dict1, aligned_dict2
    Raises:
        ValueError if the dicts don't share identical keys
    """
    if set(dict1.keys()) != set(dict2.keys()):
        missing_1 = set(dict2.keys()) - set(dict1.keys())
        missing_2 = set(dict1.keys()) - set(dict2.keys())
        raise ValueError(f"Key mismatch:\nOnly in dict1: {missing_2}\nOnly in dict2: {missing_1}")

    # Order both dicts by dict1's key order
    ordered_keys = list(dict1.keys())
    dict1_ordered = {k: dict1[k] for k in ordered_keys}
    dict2_ordered = {k: dict2[k] for k in ordered_keys}
    
    return dict1_ordered, dict2_ordered
