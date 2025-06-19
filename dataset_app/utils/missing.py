# app/utils/missing.py

import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import streamlit as st
import time
from dataset_app.utils.general import *

def extract_feature_importances(shadow_models, shadow_splits):
    """
    Computes and combines feature importances from multiple shadow models,
    supporting both XGBoost and Random Forest.

    Args:
        shadow_models: list of trained models (XGBoost or Random Forest).
        shadow_splits: list of (X_train, X_test, X_ext, y_train, y_test, y_ext) tuples.

    Returns:
        model_importances: list of pandas Series (one per model) with feature importances.
        combined_importance: pandas Series with mean importance across models.
    """
    model_importances = []

    for model, split in zip(shadow_models, shadow_splits):
        X_train = split[0]
        feature_names = X_train.columns

        model_type = check_model_type(model)

        if model_type == 'XGBoost Model':
            # XGBoost uses feature names like f0, f1...
            booster_scores = model.get_booster().get_score(importance_type='weight')
            scores = {f"f{idx}": booster_scores.get(f"f{idx}", 0.0) for idx in range(len(feature_names))}
            importance_series = pd.Series(scores.values(), index=feature_names)

        elif model_type == 'Random Forest Model':
            importance_series = pd.Series(model.feature_importances_, index=feature_names)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_importances.append(importance_series)

    combined_importance = pd.concat(model_importances, axis=1).mean(axis=1)

    return model_importances, combined_importance

def calculate_missing_stats(X_train, model, feature_importance, features_means, features_medians, strategy='mean', n_important=3, desc=""):
    """
    Calculates statistics (standard deviation, entropy, variance) of target model predictions
    on a dataset with strategically introduced missing values.

    Args:
        X_train (pd.DataFrame): The training dataset.
        target_model: The trained target model for prediction.
        feature_importance (pd.Series): Feature importance scores.
        features_means (pd.Series): Mean values of features.
        features_medians (pd.Series): Median values of features.
        strategy (str, optional): Strategy for introducing missing values. Defaults to 'mean'.
        n_important (int, optional): Number of important features to consider. Defaults to 3.

    Returns:
        tuple: A tuple containing lists of standard deviations, entropies, and variances.
    """

    missing_train_stds = []
    missing_train_entropies = []
    missing_train_vars = []

    # Iterate over rows of X_train with tqdm
    for index, row in stqdm(X_train.iterrows(), total=len(X_train), desc=desc):
        # Function to create the set with missing values
        missing_set = create_missing_set(row, feature_importance, features_means, features_medians, strategy, n_important)
        missing_set_df = pd.DataFrame(missing_set, columns=X_train.columns)

        # Predict the house price for modified set
        missing_set_preds = model.predict(missing_set_df)

        # Calculate statistics and store in their respective lists
        missing_train_stds.append(np.std(missing_set_preds))
        missing_train_entropies.append(entropy(missing_set_preds))
        missing_train_vars.append(np.var(missing_set_preds))

    return missing_train_stds, missing_train_entropies, missing_train_vars

def process_missing_row(args):
    index, row, model_id, strategy, n_important = args
    model = get_global_shadow_model(model_id - 1) if model_id > 0 else get_global_target_model()
    importance = get_global_feature_importance(model_id - 1 if model_id > 0 else 0)
    means = get_global_feature_means(model_id - 1 if model_id > 0 else 0)
    medians = get_global_feature_medians(model_id - 1 if model_id > 0 else 0)
    missing_set = create_missing_set(row, importance, means, medians, strategy=strategy, n_important=n_important)
    df = pd.DataFrame(missing_set, columns=row.index)
    preds = model.predict(df)
    return np.std(preds), entropy(preds), np.var(preds)


def create_missing_set(original_row, feature_importance, features_means, features_medians, strategy='mean', n_important=5):
    """
    Create a missing set for a certain record where each record equals
    the original record but one feature, which is set to zero/mean/median.

    Parameters:
    - original_row: Pandas Series representing the original record.
    - feature_importance: Pandas Series representing the feature importance scores.
    - features_means: Series of mean values per feature (used if strategy='mean').
    - features_medians: Series of median values per feature (used if strategy='median').
    - strategy: Which strategy to use ('zero', 'mean', 'median').
    - n_important: Number of top important features to set as missing.

    Returns:
    - missing_set: List of Pandas Series representing the missing set.
    """
    feature_importance_list = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in feature_importance_list]
    missing_set = [original_row]

    for indice in sorted_indices[:n_important]:
        missing_row = original_row.copy()

        if strategy == 'zero':
            missing_row[indice] = 0
        elif strategy == 'mean':
            missing_row[indice] = features_means[indice]
        elif strategy == 'median':
            if features_medians is None:
                raise ValueError("features_medians must be provided for strategy='median'")
            missing_row[indice] = features_medians[indice]

        missing_set.append(missing_row)

    return missing_set

def parallel_process_missing_rows(X, model_id, strategy, n_important, desc, max_workers=4):
    args_list = [(index, row, model_id, strategy, n_important) for index, row in X.iterrows()]
    total = len(args_list)
    results = [None] * total
    start_time = time.time()

    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(f"{desc} ({total} records):")
    with progress_col:
        progress = st.progress(0)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_missing_row, args): i for i, args in enumerate(args_list)}

        for completed_idx, future in enumerate(as_completed(futures)):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"⚠️ Error processing row {i}: {e}")
                results[i] = (0.0, 0.0, 0.0)
            progress.progress(int((completed_idx + 1) / total * 100))

    duration = time.time() - start_time
    #st.toast(f"✔ Missing stats processed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_missing_rows took {duration:.2f} seconds")
    return results