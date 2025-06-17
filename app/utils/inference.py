# app/utils/inference.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def plot_metrics_three_sets(
    set1: tuple[np.ndarray, np.ndarray, np.ndarray],
    set2: tuple[np.ndarray, np.ndarray, np.ndarray],
    set3: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: tuple[str, str, str] = ('Set 1', 'Set 2', 'Set 3')
):
    def compute_the_metrics(y_true, y_pred, y_proba):
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        tpr_at_fpr_10 = compute_tpr_at_fpr(y_true, y_proba, target_fpr=0.1)
        return accuracy, auc, tpr_at_fpr_10

    metrics1 = compute_the_metrics(*set1)
    metrics2 = compute_the_metrics(*set2)
    metrics3 = compute_the_metrics(*set3)

    table_data = [
        ['Accuracy', f"{metrics1[0]:.0%}", f"{metrics2[0]:.0%}", f"**{metrics3[0]:.0%}**"],
        ['AUC', f"{metrics1[1]:.0%}", f"{metrics2[1]:.0%}", f"**{metrics3[1]:.0%}**"],
        ['TPR@FPR=10%', f"{metrics1[2]:.0%}", f"{metrics2[2]:.0%}", f"**{metrics3[2]:.0%}**"],
    ]

    column_labels = ['', labels[0], labels[1], labels[2]]
    df = pd.DataFrame(table_data, columns=column_labels)
    return df

def compute_tpr_at_fpr(y_true, y_scores, target_fpr=0.1):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    if max(fpr) < target_fpr:
        return np.nan
    return np.interp(target_fpr, fpr, tpr)