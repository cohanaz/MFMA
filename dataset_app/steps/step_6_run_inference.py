# step_6_run_inference.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
from dataset_app.utils.inference import compute_tpr_at_fpr
from dataset_app.utils.general import *
from dataset_app.utils.augmentations import *
from dataset_app.utils.missing import *
from dataset_app.utils.ens_var import *
from sklearn.metrics import accuracy_score, roc_auc_score
import time

USE_JOBLIB = True  # שנה ל-False אם אתה רוצה לחזור ל-ProcessPoolExecutor

def run():
    st.subheader("Step 6: Run Dataset-level Inference Attack")

    st.markdown("### 📊 Attack Summary Table")

    if "dataset_attack_results" not in st.session_state:
        st.session_state.dataset_attack_results = pd.DataFrame(
            columns=["Target Model", "Owned", "#Detected Members", "Mean Proba", "Accuracy", "AUC", "TPR@FPR=0.1"]
        )
    st.dataframe(st.session_state.dataset_attack_results, hide_index=True)

    if not st.session_state.get("inference_model_trained"):
        st.warning("Please train the inference model in Step 5 first.")
        return

    if not all(k in st.session_state for k in ["target_models", "dataset_owned", "dataset_external"]):
        st.warning("Missing target models or ownership information from previous step.")
        return

    if "dataset_attack_results" not in st.session_state:
        st.session_state.dataset_attack_results = pd.DataFrame(
            columns=["Target Model", "Owned", "#Detected Members", "Accuracy", "AUC", "TPR@FPR=0.1"]
        )

    clf = st.session_state.get("inference_model")
    #set_global_target_models(st.session_state.target_models)
    #set_global_target_splits(st.session_state.target_splits)

    #st.session_state.target_model_importances, st.session_state.target_combined_importance = extract_feature_importances(st.session_state.target_models, st.session_state.target_splits)

    columns = ['index', 'aug_preds_var', 'aug_preds_range', 'aug_preds_diff'] + [f'aug_pred_{i}' for i in range(st.session_state.augmentation_count)]

    if "target_model_idx" not in st.session_state:
        st.session_state.target_model_idx = 0

    run_all = st.session_state.get("run_all_mode", False)

    if st.session_state.target_model_idx < len(st.session_state.target_models):
        col = st.columns([1, 1, 1, 1])
        with col[1]:
            if st.button(f"✅ Run Model {st.session_state.target_model_idx+1}", use_container_width=True):
                st.session_state.run_all_mode = False
                st.session_state.run_next_model = True
                st.rerun()

        with col[2]:
            if st.button("▶️ Run All", use_container_width=True):
                st.session_state.run_all_mode = True
                st.session_state.run_next_model = True
                st.rerun()
    else:
        # כל המודלים חושבו
        run_all = False
        st.session_state.run_next_model = False
        st.session_state.run_all_mode = False
        st.success("✅ All target models have been processed!")

    if run_all or st.session_state.get("run_next_model"):
        i = st.session_state.target_model_idx
        target_model = st.session_state.target_models[i]

        is_owned = i < int(len(st.session_state.target_models) * st.session_state.owned_model_ratio)
        X_train, X_test, X_ext, y_train, y_test, y_ext = st.session_state.target_splits[i]
        target_X_train = X_train
        target_y_train = y_train
        target_X_test = X_test
        target_y_test = y_test

        train_t_preds = target_model.predict(target_X_train)
        test_t_preds = target_model.predict(target_X_test)
        train_t_errors = abs(train_t_preds - target_y_train)
        test_t_errors = abs(test_t_preds - target_y_test)

        #st.toast('Errors & perdictions extracted successfully!')

        for noise in st.session_state.noise_levels:
            noise_str = str(noise).replace('.', '')
            while noise_str.endswith('0') and len(noise_str) > 1:
                noise_str = noise_str[:-1]

            st.markdown(f"**Processing augmentations @ noise={noise}:**")

            results_train = parallel_process_rows_flexible(
                X=target_X_train,
                feature_scale=noise,
                model_id=0,
                model=target_model,
                augmented_records=st.session_state.augmentation_count,
                desc="Target model train set",
                max_workers=st.session_state.max_workers,
                batch_size=16,
                use_joblib=USE_JOBLIB
            )
            st.session_state[f"aug_train_t_{noise_str}"] = pd.DataFrame(results_train, columns=columns)

            results_test = parallel_process_rows_flexible(
                X=target_X_test,
                feature_scale=noise,
                model_id=0,
                model=target_model,
                augmented_records=st.session_state.augmentation_count,
                desc="Target model test set",
                max_workers=st.session_state.max_workers,
                batch_size=16,
                use_joblib=USE_JOBLIB
            )
            st.session_state[f"aug_test_t_{noise_str}"] = pd.DataFrame(results_test, columns=columns)

        #st.toast('Augmentation features extracted successfully!')

        st.markdown(f"**Processing missing features:**")

        features_t_means = target_X_train.mean()
        features_t_medians = target_X_train.median()

        #set_global_feature_importances(st.session_state.target_model_importances[i])
        #set_global_feature_means_list([features_t_means])
        #set_global_feature_medians_list([features_t_medians])

        missing_train_t_stds, missing_train_t_entropies, missing_train_t_vars = zip(*parallel_process_missing_rows_joblib(
            X=target_X_train,
            model=target_model,
            importance=st.session_state.combined_importance,
            means=features_t_means,
            medians=features_t_medians,
            strategy=st.session_state.missing_strategy,
            n_important=st.session_state.n_missing,
            desc="Target model train set",
            max_workers=st.session_state.max_workers
        ))

        missing_test_t_stds, missing_test_t_entropies, missing_test_t_vars = zip(*parallel_process_missing_rows_joblib(
            X=target_X_test,
            model=target_model,
            importance=st.session_state.combined_importance,
            means=features_t_means,
            medians=features_t_medians,
            strategy=st.session_state.missing_strategy,
            n_important=st.session_state.n_missing,
            desc="Target model test set",
            max_workers=st.session_state.max_workers
        ))

        #st.toast('Missing features extracted successfully!')

        st.markdown(f"**Processing ensemble variation features:**")
        est_func = parallel_estimators_metrics if st.session_state.target_model_type == "XGBoost" else calculate_tree_stats
        ens_var_train_metric_1, ens_var_train_metric_2 = est_func(model=target_model, X=target_X_train, desc="Target model train set", max_workers=st.session_state.max_workers)
        ens_var_test_metric_1, ens_var_test_metric_2 = est_func(model=target_model, X=target_X_test, desc="Target model test set", max_workers=st.session_state.max_workers)

        #st.toast('Ensemble features extracted successfully!')

        data_test_dict = {
            'prediction': list(train_t_preds) + list(test_t_preds),
            'error': list(train_t_errors) + list(test_t_errors),
            'membership': (
                list(np.ones(len(target_y_train))) + list(np.zeros(len(target_y_test)))
                if is_owned else
                list(np.zeros(len(target_y_train) + len(target_y_test)))
            ),
            'missing_preds_entropies': list(missing_train_t_entropies) + list(missing_test_t_entropies),
            'missing_preds_vars': list(missing_train_t_vars) + list(missing_test_t_vars),
            'ens_var_metric_1': list(ens_var_train_metric_1) + list(ens_var_test_metric_1),
            'ens_var_metric_2': list(ens_var_train_metric_2) + list(ens_var_test_metric_2)
        }

        if 1.0 in st.session_state.noise_levels:
            data_test_dict['aug_preds_var_1'] = list(st.session_state.aug_train_t_1['aug_preds_var']) + list(st.session_state.aug_test_t_1['aug_preds_var'])
            data_test_dict['aug_preds_range_1'] = list(st.session_state.aug_train_t_1['aug_preds_range']) + list(st.session_state.aug_test_t_1['aug_preds_range'])
            data_test_dict['aug_preds_diff_1'] = list(st.session_state.aug_train_t_1['aug_preds_diff']) + list(st.session_state.aug_test_t_1['aug_preds_diff'])
        if 0.1 in st.session_state.noise_levels:
            data_test_dict['aug_preds_var_01'] = list(st.session_state.aug_train_t_01['aug_preds_var']) + list(st.session_state.aug_test_t_01['aug_preds_var'])
            data_test_dict['aug_preds_range_01'] = list(st.session_state.aug_train_t_01['aug_preds_range']) + list(st.session_state.aug_test_t_01['aug_preds_range'])
            data_test_dict['aug_preds_diff_01'] = list(st.session_state.aug_train_t_01['aug_preds_diff']) + list(st.session_state.aug_test_t_01['aug_preds_diff'])
        if 0.01 in st.session_state.noise_levels:
            data_test_dict['aug_preds_var_001'] = list(st.session_state.aug_train_t_001['aug_preds_var']) + list(st.session_state.aug_test_t_001['aug_preds_var'])
            data_test_dict['aug_preds_range_001'] = list(st.session_state.aug_train_t_001['aug_preds_range']) + list(st.session_state.aug_test_t_001['aug_preds_range'])
            data_test_dict['aug_preds_diff_001'] = list(st.session_state.aug_train_t_001['aug_preds_diff']) + list(st.session_state.aug_test_t_001['aug_preds_diff'])

        # Map each (train_df, test_df) to its corresponding noise label
        aug_dfs = []
        for suffix in ['1', '01', '001']:
            train_key = f'aug_train_t_{suffix}'
            test_key = f'aug_test_t_{suffix}'    
            if train_key in st.session_state and test_key in st.session_state:
                aug_dfs.append((st.session_state[train_key], st.session_state[test_key], suffix))

        # Add all prediction features with renamed columns
        for aug_train_df, aug_test_df, noise_label in aug_dfs:
            for col in aug_train_df.columns:
                if col.startswith('aug_pred_'):
                    # Extract the number from 'aug_pred_0' → '0'
                    col_suffix = col.replace('aug_pred_', '')
                    new_col = f'aug_pred_{noise_label}_{col_suffix}'
                    data_test_dict[new_col] = list(aug_train_df[col]) + list(aug_test_df[col])

        df_test = pd.DataFrame({k: v for k, v in data_test_dict.items() if k in st.session_state.selected_features})
        df_test["membership"] = data_test_dict["membership"]
        data_test = df_test.replace([np.inf, -np.inf], 0)

        feature_order = [key for key in st.session_state.data_attack_dict.keys()]
        data_test = data_test.reindex(columns=feature_order)

        X_attack = data_test.drop('membership', axis=1)
        y_attack = data_test['membership']

        probas = clf.predict_proba(X_attack)[:, 1]
        threshold = 0.7
        preds = (probas >= threshold).astype(int)
        detected = preds.sum()
        total = len(X_attack)
        percent = 100 * detected / total if total > 0 else 0

        acc = accuracy_score(y_attack, preds)
        auc = roc_auc_score(y_attack, probas)
        tpr_fpr = compute_tpr_at_fpr(y_attack, probas, target_fpr=0.1)
        mean_proba = float(np.mean(probas))

        new_row = {
            "Target Model": i+1,
            "Owned": is_owned,
            "#Detected Members": f"{int(detected)}/{total} ({percent:.0f}%)",
            "Mean Proba": round(mean_proba, 2),
            'std_proba': np.std(probas),
            'proba_entropy': entropy([np.mean(probas), 1 - np.mean(probas)]),
            'skewness': skew(probas),
            'kurtosis': kurtosis(probas),
            'IQR': np.percentile(probas, 75) - np.percentile(probas, 25),
            'percent_above_05': np.mean(probas > 0.5),
            'percent_above_07': np.mean(probas > 0.7),
            'percent_below_03': np.mean(probas < 0.3),
            "Accuracy": round(acc, 2),
            "AUC": round(auc, 2),
            "TPR@FPR=0.1": round(tpr_fpr, 2)
        }
        st.session_state.dataset_attack_results = pd.concat([
            st.session_state.dataset_attack_results,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        st.session_state.target_model_idx += 1
        st.session_state.run_next_model = False

        members_proba = probas[y_attack == 1]
        non_members_proba = probas[y_attack == 0]
        fig, ax = plt.subplots()
        ax.hist(non_members_proba, bins=20, alpha=0.6, label='Non-Members', color='gray', edgecolor='black')
        ax.hist(members_proba, bins=20, alpha=0.6, label='Members', color='blue', edgecolor='black')
        ax.set_title(f"Prediction Probabilities by Class - Target Model {i}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
    
        confirm_col = st.columns([2, 1, 2])[1]
        with confirm_col:
            if st.button("✅ Confirm", use_container_width=True):
                st.rerun()

        if run_all:
            time.sleep(0.25)
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_step -= 1
            st.rerun()
