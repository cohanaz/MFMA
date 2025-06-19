# app/utils/ens_var.py

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import entropy
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import streamlit as st
import time
from app.utils.general import *

def compute_estimators_metrics(X, model, alpha=0.01, window_prcnt=0.1, desc="", max_workers=None):
    """
    Computes per-record tree contribution metrics for XGBRegressor:
    - Ratio of positive vs. negative contributions
    - Standard deviation of contributions
    - Direction consistency (how many trees contributed in same direction)
    - Convergence point of prediction

    Parameters:
        model (xgb.XGBRegressor): Trained model
        X_data (pd.DataFrame or np.array): Input data
        alpha (float): Convergence sensitivity threshold

    Returns:
        Tuple of lists: (std_contribs, directional_consistencies, convergence_indexs)
    """
    booster = model.get_booster()
    num_trees = booster.num_boosted_rounds()
    std_contribs = []
    directional_consistencies = []
    convergence_indexs = []

    start_time = time.time()

    for i in stqdm(range(len(X)), desc=desc):
        dmatrix = xgb.DMatrix(X.iloc[[i]] if isinstance(X, pd.DataFrame) else X[i:i+1])

        predictions = np.array([
            booster.predict(dmatrix, iteration_range=(0, t + 1))[0]
            for t in range(num_trees)
        ])

        contributions = np.diff(predictions, prepend=0)
        num_pos = np.sum(contributions > 0)
        num_neg = np.sum(contributions < 0)

        pos_ratio = num_pos / num_trees
        neg_ratio = num_neg / num_trees
        std_contrib = np.std(contributions)
        directional_consistency = max(num_pos, num_neg) / num_trees

        final_pred = predictions[-1]
        threshold = alpha * abs(final_pred) if final_pred != 0 else alpha

        # הפרשים בין תחזיות עוקבות
        diffs = np.abs(np.diff(predictions))

        # חפש רצף של window צעדים בהם כל ההפרשים קטנים מהסף
        window = int(window_prcnt * num_trees)
        converged = False
        for t in range(num_trees - window):
            if np.all(diffs[t:t+window] < threshold):
                convergence_ratio = (t + 1) / num_trees  # +1 כי diff מתחיל אחרי העץ הראשון
                converged = True
                break
        if not converged:
            convergence_ratio = 1.0

        std_contribs.append(std_contrib)
        directional_consistencies.append(directional_consistency)
        convergence_indexs.append(convergence_ratio)

    elapsed = time.time() - start_time
    print(f"✅ {desc} completed in {elapsed:.2f} seconds")

    return directional_consistencies, convergence_indexs

def compute_single_booster_metrics(index, row, model, alpha, window_prcnt):
    try:
        booster = model.get_booster()
        num_trees = booster.num_boosted_rounds()

        dmatrix = xgb.DMatrix(pd.DataFrame([row]) if isinstance(row, pd.Series) else row.reshape(1, -1))
        predictions = np.array([
            booster.predict(dmatrix, iteration_range=(0, t + 1))[0]
            for t in range(num_trees)
        ])

        contributions = np.diff(predictions, prepend=0)
        num_pos = np.sum(contributions > 0)
        num_neg = np.sum(contributions < 0)

        directional_consistency = max(num_pos, num_neg) / num_trees

        final_pred = predictions[-1]
        threshold = alpha * abs(final_pred) if final_pred != 0 else alpha
        diffs = np.abs(np.diff(predictions))
        window = int(window_prcnt * num_trees)

        for t in range(num_trees - window):
            if np.all(diffs[t:t+window] < threshold):
                convergence_ratio = (t + 1) / num_trees
                break
        else:
            convergence_ratio = 1.0

        return (index, directional_consistency, convergence_ratio)

    except Exception as e:
        print(f"❗ Error in row {index}: {e}")
        return (index, 0.0, 1.0)

def parallel_estimators_metrics(model, X, alpha=0.01, window_prcnt=0.1, max_workers=4, desc=""):
    start_time = time.time()

    with st.spinner(f"⏳ {desc} — computing in parallel..."):
        tasks = [
            delayed(compute_single_booster_metrics)(i, row, model, alpha, window_prcnt)
            for i, row in X.iterrows()
        ]

        results = Parallel(n_jobs=max_workers)(tasks)

        results.sort(key=lambda x: x[0])
        directional_consistencies = [r[1] for r in results]
        convergence_ratios = [r[2] for r in results]

    elapsed = time.time() - start_time
    st.write(f"✅ {desc} completed in {elapsed:.2f} seconds")
    print(f"✅ {desc} completed in {elapsed:.2f} seconds")

    return directional_consistencies, convergence_ratios


def calculate_tree_stats(model, X, desc, max_workers=None):
    """
    Calculates statistics (standard deviation, entropy, variance) of predictions
    from individual trees in an ensemble model.

    Args:
        model: The trained ensemble model (e.g., RandomForestRegressor).
        X (pd.DataFrame): The dataset to predict on.

    Returns:
        tuple: A tuple containing lists of standard deviations, entropies, and variances.
    """
    st.markdown(desc)

    # Get predictions from individual trees
    tree_predictions = np.array([tree.predict(X.values) for tree in model.estimators_])

    # Calculate statistics across predictions for each sample
    stds = np.std(tree_predictions, axis=0)
    vars_ = np.var(tree_predictions, axis=0)
    
    return stds, vars_