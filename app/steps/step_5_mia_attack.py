# step_5_mia_attack.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from app.utils.inference import plot_metrics_three_sets, compute_tpr_at_fpr


def run():
    
    st.subheader("Step 5: Train & Test Inference Model")

    # Group features by prefix
    from collections import defaultdict

    grouped = defaultdict(list)
    for key in st.session_state.data_attack_dict.keys():
        if key in ["index", "membership"]:
            continue
        if key.startswith("aug_pred_"):
            grouped["AUG_preds"].append(key)
        elif key.startswith("aug_"):
            grouped["AUG_stats"].append(key)
        elif key.startswith("ens_"):
            grouped["Ens_var"].append(key)
        else:
            prefix = key.split("_")[0]
            grouped[prefix].append(key)
          

    feature_groups = list(grouped.keys())
    selected_groups = st.multiselect("Select feature groups to train the inference model:", options=feature_groups, default=feature_groups)

    selected_features = []
    for group in selected_groups:
        selected_features.extend(grouped[group])

    if selected_features:
        # Prepare data
        df_attack = pd.DataFrame({k: v for k, v in st.session_state.data_attack_dict.items() if k in selected_features})
        df_attack["membership"] = st.session_state.data_attack_dict["membership"]

        df_test = pd.DataFrame({k: v for k, v in st.session_state.data_test_dict.items() if k in selected_features})
        df_test["membership"] = st.session_state.data_test_dict["membership"]
    else:
        st.warning("Please select at least one feature group.")


    centered_col = st.columns([1, 3, 1])[1]
    with centered_col:
        clicked = st.button("Train Inference Model", use_container_width=True)

    if clicked:
        # Prepare data
        data_attack = df_attack.replace([np.inf, -np.inf], 0)
        data_test = df_test.replace([np.inf, -np.inf], 0)

        Xt_attack_test = data_test.drop('membership', axis=1)
        yt_attack_test = data_test['membership']

        X_attack = data_attack.drop('membership', axis=1)
        y_attack = data_attack['membership']

        X_train_att, X_test_att, y_train_att, y_test_att = train_test_split(X_attack, y_attack, test_size=0.1, stratify=y_attack, random_state=42)

        # Train model
        st.markdown("**Training inference model...**")
        #Best Hyperparameters: {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.8}
        clf = xgb.XGBClassifier(colsample_bytree=0.8, gamma=0.1, learning_rate=0.01, max_depth=5, min_child_weight=1, subsample= 0.8, n_estimators=400, random_state=42)
        clf.fit(X_train_att, y_train_att)

        # Predict
        pred_shadow_train = clf.predict(X_train_att)
        pred_shadow_test = clf.predict(X_test_att)
        pred_target = clf.predict(Xt_attack_test)
        proba_shadow_train = clf.predict_proba(X_train_att)[:, 1]
        proba_shadow_test = clf.predict_proba(X_test_att)[:, 1]
        proba_target = clf.predict_proba(Xt_attack_test)[:,1]

        st.session_state.infrence_results = plot_metrics_three_sets(
            (y_train_att, pred_shadow_train, proba_shadow_train),
            (y_test_att, pred_shadow_test, proba_shadow_test),
            (yt_attack_test, pred_target, proba_target),
            labels=('Shadow train set', 'Shadow test set', '**Target set**')
        )

        def compute_metrics(y_true, y_pred, y_proba):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_proba),
                "TPR@FPR=0.1": compute_tpr_at_fpr(y_true, y_proba, target_fpr=0.1)
            }

        st.session_state.metrics_target = compute_metrics(yt_attack_test, pred_target, proba_target)
    
    # Initialize summary table if not exists
    if "inference_results_table" not in st.session_state:
        results_cols = []
        for group in feature_groups:
            results_cols.append(f"{group}")
        results_cols.extend(["Accuracy", "AUC", "TPR@FPR=0.1"]) 
        st.session_state.inference_results_table = pd.DataFrame(columns=results_cols)

    # אתחול המשתנה אם צריך
    if "metrics_target" not in st.session_state:
        st.session_state.metrics_target = {}

    # Button to add a row
    if st.session_state.metrics_target != {}:
        #st.dataframe(st.session_state.infrence_results, use_container_width=True)
        st.table(st.session_state.infrence_results)
        col_add = st.columns([1, 2, 1])[1]
        with col_add:
            if st.button("📥 Add Target Set Results to Summary Table", use_container_width=True):
                st.session_state.add_row_requested = True

    if st.session_state.get("add_row_requested", False):
        row = {}
        for group in feature_groups:
            row[f"{group}"] = group in selected_groups
        for metric_name, value in st.session_state.metrics_target.items():
            row[metric_name] = round(value, 2)

        st.write("Row to add:", row)

        st.session_state.inference_results_table = pd.concat(
            [st.session_state.inference_results_table, pd.DataFrame([row])],
            ignore_index=True
        )

        st.toast("✅ Target results added to summary table!")
        st.session_state.metrics_target = {}
        st.session_state.add_row_requested = False
        st.rerun()

    # Show the table
    #if not st.session_state.inference_results_table.empty:
    st.markdown("### 📊 Target Results Summary")

    # #if not st.session_state.inference_results_table.empty:
    df = st.session_state.inference_results_table

    column_config = {}

    for i, col in enumerate(df.columns):
        column_config[col] = st.column_config.Column(width="small")

    # הצגת הטבלה
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)

    # אתחול הדגל אם צריך
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    # כפתור איפוס ראשי
    col_reset = st.columns([1, 2, 1])[1]
    with col_reset:
        if st.button("🔄 Reset Table", use_container_width=True):
            st.session_state.confirm_reset = True

    # הצגת אזהרה וכפתורי אישור אם נלחץ reset
    if st.session_state.confirm_reset:
        col1 = st.columns([1, 2, 1])[1]
        with col1:
            st.warning("⚠️ Are you sure you want to delete this item?")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("✅ Yes, delete", use_container_width=True):
                    st.session_state.inference_results_table = pd.DataFrame(
                        columns=st.session_state.inference_results_table.columns
                    )
                    st.session_state.metrics_target = {}
                    st.session_state.add_row_requested = False
                    st.session_state.confirm_reset = False
                    st.toast("🗑️ Table reset successfully!")
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_step -= 1
            st.session_state.metrics_target = {}
            st.rerun()

