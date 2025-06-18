# step_4_feature_extraction.py

import multiprocessing
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

from app.utils.feature_extraction import *

USE_JOBLIB = True  # שנה ל-False אם אתה רוצה לחזור ל-ProcessPoolExecutor

def run():
    st.subheader("Step 4: Compute Attack Features")

    if "feature_stage" not in st.session_state:
        st.session_state.feature_stage = 0

    if "feature_completed" not in st.session_state:
        st.session_state.feature_completed = [False] * 4

    feature_sequence = ["Error", "Augmentations", "Missing", "Ensemble Variation", "Combining"]
    current = st.session_state.feature_stage
    current_label = feature_sequence[current]
    
    # Display feature progress
    completed = st.session_state.feature_completed
    substeps = ["Error", "Augmentations", "Missing", "Ensemble Variation"]
    #st.markdown("#### Feature Progress:")
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    with progress_col1:
        label = f"✅ {substeps[0]}" if completed[0] else f"🔲 {substeps[0]}"
        st.markdown(label)
    with progress_col2:
        label = f"✅ {substeps[1]}" if completed[1] else f"🔲 {substeps[1]}"
        st.markdown(label)
    with progress_col3:
        label = f"✅ {substeps[2]}" if completed[2] else f"🔲 {substeps[2]}"
        st.markdown(label)
    with progress_col4:
        label = f"✅ {substeps[3]}" if completed[3] else f"🔲 {substeps[3]}"
        st.markdown(label)

    st.markdown(
        f"<h4 style='text-align: center;'>Feature Group: {current_label}</h3>",
        unsafe_allow_html=True
    )

    if current_label == "Error":
        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            if st.button("Compute Error Features", use_container_width=True):
                progress = st.progress(0)
                with st.spinner("Computing Error Features..."):
                    progress.progress(10)
                    st.session_state.train_t_preds = st.session_state.target_model.predict(st.session_state.target_X_train)
                    st.session_state.test_t_preds = st.session_state.target_model.predict(st.session_state.target_X_test)

                    st.session_state.train_s_preds_list = []
                    st.session_state.test_s_preds_list = []
                
                    for model, (X_train, X_test, _, _, _, _) in zip(st.session_state.shadow_models, st.session_state.shadow_splits):
                        st.session_state.train_s_preds = model.predict(X_train)
                        st.session_state.test_s_preds = model.predict(X_test)
                        st.session_state.train_s_preds_list.append(st.session_state.train_s_preds)
                        st.session_state.test_s_preds_list.append(st.session_state.test_s_preds)

                    st.session_state.train_t_errors = abs(st.session_state.train_t_preds - st.session_state.target_y_train)
                    st.session_state.test_t_errors = abs(st.session_state.test_t_preds - st.session_state.target_y_test)

                    st.session_state.train_s_errors_list = []
                    st.session_state.test_s_errors_list = []

                    for (train_preds, test_preds), (_, _, _, y_train, y_test, _) in zip(zip(st.session_state.train_s_preds_list, st.session_state.test_s_preds_list), st.session_state.shadow_splits):
                        st.session_state.train_errors = abs(train_preds - y_train)
                        st.session_state.test_errors = abs(test_preds - y_test)
                        st.session_state.train_s_errors_list.append(st.session_state.train_errors)
                        st.session_state.test_s_errors_list.append(st.session_state.test_errors)

                    st.session_state.model_importances, st.session_state.feature_importance_t = extract_feature_importances(st.session_state.shadow_models, st.session_state.shadow_splits)
                    st.session_state.features_t_means = st.session_state.target_X_train.mean()
                    st.session_state.features_t_medians = st.session_state.target_X_train.median()
                    st.session_state.features_s_means_list = []
                    st.session_state.features_s_medians_list = []

                    for split in st.session_state.shadow_splits:
                        X_train = split[0]  # X_train is the first element in the tuple
                        st.session_state.features_s_means = X_train.mean()
                        st.session_state.features_s_means_list.append(st.session_state.features_s_means)
                        st.session_state.features_s_medians = X_train.median()
                        st.session_state.features_s_medians_list.append(st.session_state.features_s_medians)

                    progress.progress(100)
                    st.session_state.feature_completed[0] = True
                    st.session_state.feature_stage += 1
                    st.rerun()

    elif current_label == "Augmentations":

        label_col, select_col = st.columns([2, 4])
        with label_col:
            st.markdown("Select noise levels:")
        with select_col:
            noise_levels = st.multiselect(
                label="", 
                options=[1.0, 0.1, 0.01], 
                default=[1.0, 0.1]
            )
            st.session_state.noise_levels = noise_levels

        label_col, select_col = st.columns([2, 4])
        with label_col:
            st.markdown("Select number of augmentations per sample:")
        with select_col:
            augmentation_count = option_menu(
                menu_title=None,
                options=[4, 8, 12, 16, 20, 24],
                orientation="horizontal",
                default_index=1,
                styles={
                    "container": {"padding": "0!important", "background-color": "#f9f9f9"},
                    "icon": {"color": "white", "font-size": "0px"},
                    "nav-link": {
                        "font-size": "16px",
                        "margin": "0px",
                        "padding": "10px 16px",
                        "border-radius": "6px",
                        "color": "black",
                        "text-align": "center",
                    },
                    "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
                }
            )

        columns = ['index', 'aug_preds_var', 'aug_preds_range', 'aug_preds_diff'] + [f'aug_pred_{i}' for i in range(augmentation_count)]
        
        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            clicked = st.button("Compute Augmentation Features", use_container_width=True)

        if clicked:

            num_cpus = multiprocessing.cpu_count()
            st.toast(f"Using {num_cpus} CPU cores for parallel processing")
            st.session_state.max_workers = num_cpus

            set_global_target_model(st.session_state.target_model)
            set_global_shadow_models(st.session_state.shadow_models)
            set_global_shadow_splits(st.session_state.shadow_splits)

            for noise in noise_levels:
                noise_str = str(noise).replace('.', '')
                while noise_str.endswith('0') and len(noise_str) > 1:
                    noise_str = noise_str[:-1]

                st.markdown(f"**Processing augmentations @ noise={noise}**")

                # Target model: train
                results_train = parallel_process_rows_flexible(
                    X=st.session_state.target_X_train,
                    feature_scale=noise,
                    model_id=0,
                    model=st.session_state.target_model,
                    augmented_records=augmentation_count,
                    desc="Target model train set",
                    max_workers=st.session_state.max_workers,
                    use_joblib=USE_JOBLIB  # משתנה שהגדרת למעלה
                )
                st.session_state[f"aug_train_t_{noise_str}"] = pd.DataFrame(results_train, columns=columns)

                # Target model: test
                results_test = parallel_process_rows_flexible(
                    X=st.session_state.target_X_test,
                    feature_scale=noise,
                    model_id=0,
                    model=st.session_state.target_model,
                    augmented_records=augmentation_count,
                    desc="Target model test set",
                    max_workers=st.session_state.max_workers,
                    use_joblib=USE_JOBLIB  # משתנה שהגדרת למעלה
                )
                st.session_state[f"aug_test_t_{noise_str}"] = pd.DataFrame(results_test, columns=columns)

                # Shadow models
                st.session_state[f"augmented_results_{noise_str}"] = build_augmented_feature_dfs(
                    shadow_models=st.session_state.shadow_models,
                    shadow_splits=st.session_state.shadow_splits,
                    feature_scale=noise,
                    columns=columns,
                    noise_label=noise_str,
                    augmented_records=augmentation_count,
                    max_workers=st.session_state.max_workers,
                    use_joblib=USE_JOBLIB
                )

            st.session_state.aug_count = augmentation_count
            st.session_state.feature_completed[1] = True
            st.session_state.feature_stage += 1
            st.rerun()

    elif current_label == "Missing":        
        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            strategy = st.radio("**Choose strategy:**", ["mean", "median", "zero"], horizontal=True)
            n_important = st.slider("Choose number of missing features:", min_value=1, max_value=5, value=3)
            
            clicked = st.button("Compute Missing Value Features", use_container_width=True)
                
        if clicked:

            set_global_feature_importances([st.session_state.feature_importance_t] + st.session_state.model_importances)
            set_global_feature_means_list([st.session_state.features_t_means] + st.session_state.features_s_means_list)
            set_global_feature_medians_list([st.session_state.features_t_medians] + st.session_state.features_s_medians_list)
            
            st.session_state.missing_train_t_stds, st.session_state.missing_train_t_entropies, st.session_state.missing_train_t_vars = zip(*parallel_process_missing_rows(
                X=st.session_state.target_X_train,
                model_id=0,
                strategy=strategy,
                n_important=n_important,
                desc="target model train set",
                max_workers=st.session_state.max_workers
            ))

            st.session_state.missing_test_t_stds, st.session_state.missing_test_t_entropies, st.session_state.missing_test_t_vars = zip(*parallel_process_missing_rows(
                X=st.session_state.target_X_test,
                model_id=0,
                strategy=strategy,
                n_important=n_important,
                desc="target model test set",
                max_workers=st.session_state.max_workers
            ))

            st.session_state.missing_train_stats = []
            st.session_state.missing_test_stats = []

            for model_idx, split in enumerate(st.session_state.shadow_splits):
                X_train, X_test = split[0], split[1]

                train_stats = parallel_process_missing_rows(
                    X=X_train,
                    model_id=model_idx + 1,
                    strategy=strategy,
                    n_important=n_important,
                    desc=f"shadow model {model_idx+1} train set",
                    max_workers=st.session_state.max_workers
                )
                st.session_state.missing_train_stats.append(list(zip(*train_stats)))

                test_stats = parallel_process_missing_rows(
                    X=X_test,
                    model_id=model_idx + 1,
                    strategy=strategy,
                    n_important=n_important,
                    desc=f"shadow model {model_idx+1} test set",
                    max_workers=st.session_state.max_workers
                )
                st.session_state.missing_test_stats.append(list(zip(*test_stats)))

            st.session_state.missing_strategy = strategy
            st.session_state.n_missing = n_important
            st.session_state.feature_completed[2] = True
            st.session_state.feature_stage += 1
            print("Missing features computed successfully!")
            st.rerun()

    elif current_label == "Ensemble Variation":
        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            clicked = st.button("Compute Estimator Variation Features", use_container_width=True)

        if clicked:
            if st.session_state.target_model_type == "XGBoost":
                est_func = compute_estimators_metrics
            else:
                est_func = calculate_tree_stats

            st.session_state.ens_var_train_metric_1, st.session_state.ens_var_train_metric_2 = est_func(st.session_state.target_model, st.session_state.target_X_train, desc="target model train set")
            st.session_state.ens_var_test_metric_1, st.session_state.ens_var_test_metric_2 = est_func(st.session_state.target_model, st.session_state.target_X_test, desc="target model test set")

            st.session_state.ens_var_train_metrics = []
            st.session_state.ens_var_test_metrics = []

            #with st.spinner("Extracting features from shadow models..."):
            for model_idx, (model, split) in enumerate(zip(st.session_state.shadow_models, st.session_state.shadow_splits)):
                X_train, X_test = split[0], split[1]

                # Train metrics
                train_metrics = est_func(model, X_train, desc=f"shadow model {model_idx+1} train set")
                st.session_state.ens_var_train_metrics.append(train_metrics)

                # Test metrics
                test_metrics = est_func(model, X_test, desc=f"shadow model {model_idx+1} test set")
                st.session_state.ens_var_test_metrics.append(test_metrics)
                
            st.session_state.feature_completed[3] = True
            st.toast("All features computed successfully!")
            st.session_state.feature_stage += 1
            st.rerun()

    elif current_label == "Combining":
        # Show rotating spinner while combining features
        with st.spinner("Combining all features..."):
            # Initialize attack dict
            data_attack_dict = {
                'prediction': [],
                'error': [],
                'membership': [],
                'missing_preds_entropies': [],
                'missing_preds_vars': [],
                'ens_var_metric_1': [],
                'ens_var_metric_2': []
            }

            # Fill attack dict
            for i in range(st.session_state.num_shadow_models):
                data_attack_dict['prediction'] += list(st.session_state.train_s_preds_list[i]) + list(st.session_state.test_s_preds_list[i])
                data_attack_dict['error'] += list(st.session_state.train_s_errors_list[i]) + list(st.session_state.test_s_errors_list[i])
                data_attack_dict['missing_preds_entropies'] += list(st.session_state.missing_train_stats[i][1]) + list(st.session_state.missing_test_stats[i][1])
                data_attack_dict['missing_preds_vars'] += list(st.session_state.missing_train_stats[i][2]) + list(st.session_state.missing_test_stats[i][2])
                data_attack_dict['ens_var_metric_1'] += list(st.session_state.ens_var_train_metrics[i][0]) + list(st.session_state.ens_var_test_metrics[i][0])
                data_attack_dict['ens_var_metric_2'] += list(st.session_state.ens_var_train_metrics[i][1]) + list(st.session_state.ens_var_test_metrics[i][1])

                y_train = st.session_state.shadow_splits[i][3]
                y_test = st.session_state.shadow_splits[i][4]
                data_attack_dict['membership'] += list(np.ones(len(y_train))) + list(np.zeros(len(y_test)))

            for key in ["augmented_results_1", "augmented_results_01", "augmented_results_001"]:
                if key in st.session_state:
                    df = st.session_state[key]
                    for col in df.columns:
                        if col not in ['index', 'model_id', 'split']:  # skip meta
                            if col not in data_attack_dict:
                                data_attack_dict[col] = []
                            data_attack_dict[col] += list(df[col])

            # Create test dict
            data_test_dict = {
                'prediction': list(st.session_state.train_t_preds) + list(st.session_state.test_t_preds),
                'error': list(st.session_state.train_t_errors) + list(st.session_state.test_t_errors),

                # Membership labels
                'membership': list(np.ones(len(st.session_state.target_y_train))) + list(np.zeros(len(st.session_state.target_y_test))),

                # Missing features
                'missing_preds_entropies': list(st.session_state.missing_train_t_entropies) + list(st.session_state.missing_test_t_entropies),
                'missing_preds_vars': list(st.session_state.missing_train_t_vars) + list(st.session_state.missing_test_t_vars),
            }
            
            data_test_dict['ens_var_metric_1'] = list(st.session_state.ens_var_train_metric_1) + list(st.session_state.ens_var_test_metric_1)
            data_test_dict['ens_var_metric_2'] = list(st.session_state.ens_var_train_metric_2) + list(st.session_state.ens_var_test_metric_2)

            # Augmented scalar features
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

            st.session_state.data_attack_dict, st.session_state.data_test_dict = align_dicts_by_keys(data_attack_dict, data_test_dict)
            st.success("All features combined successfully!")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_step -= 1
            st.session_state.feature_stage = 0
            st.rerun()
    with col2:
        if current == len(feature_sequence) - 1:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.active_step += 1
                st.rerun()