# MFMA_app.py

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from streamlit_option_menu import option_menu
from MFMA_core import (
    compute_tpr_at_fpr, load_house_data, load_census_data,
    load_abalone_data, load_kc_house_data, load_diamonds_data,
    train_target_model, train_XGB_model, train_RF_model,
    train_shadow_models, extract_feature_importances, parallel_process_rows,
    build_augmented_feature_dfs, calculate_missing_stats, compute_estimators_metrics, calculate_tree_stats, plot_metrics_three_sets
)

st.set_page_config(page_title="MFMA", layout="centered")

# Session state initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "feature_stage" not in st.session_state:
    st.session_state.feature_stage = 0
if "feature_completed" not in st.session_state:
    st.session_state.feature_completed = [False] * 4

# Wizard-style progress bar
step_labels = ["1: Dataset", "2: Target", "3: Shadows", "4: Features", "5: Inference"]
st.markdown("### MFMA Wizard")

# Visual step bar using markdown and highlighting current step
step_bar = ""
for i, label in enumerate(step_labels):
    if i == st.session_state.active_tab:
        step_bar += f"<span style='color: white; background-color: #4CAF50; padding: 4px 10px; border-radius: 5px;'>{label}</span>"
    else:
        step_bar += f"<span style='color: #4CAF50; background-color: #E8F5E9; padding: 4px 10px; border-radius: 5px;'>{label}</span>"
    if i < len(step_labels) - 1:
        step_bar += " <b>→</b> "
st.markdown(step_bar, unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Step 1: Dataset Selection
# -------------------------
if st.session_state.active_tab == 0:
    col1, col2, col3 = st.columns([2, 1, 4])

    with col1:
        dataset_name = st.radio("**Choose a built-in dataset:**", [
            "House", "Census", "Abalone", "KC House", "Diamonds"
        ])

    with col2:
        st.markdown("""
            <div style='display: flex; justify-content: left; align-items: center; height: 100px;'>
                <strong style='font-size: 20px;'>Or</strong>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        uploaded_file = st.file_uploader("**Upload your own CSV file:**", type="csv")

    if dataset_name == "House":
        st.session_state.dataset = load_house_data()
        st.session_state.target_column = "SalePrice"
    elif dataset_name == "Census":
        st.session_state.dataset = load_census_data()
        st.session_state.target_column = "Income"
    elif dataset_name == "Abalone":
        st.session_state.dataset = load_abalone_data()
        st.session_state.target_column = "Rings"
    elif dataset_name == "KC House":
        st.session_state.dataset = load_kc_house_data()
        st.session_state.target_column = "price"
    elif dataset_name == "Diamonds":
        st.session_state.dataset = load_diamonds_data()
        st.session_state.target_column = "price"

    if uploaded_file:
        st.session_state.dataset = pd.read_csv(uploaded_file)
        st.session_state.target_column = None

    if st.session_state.dataset is not None:
        df = st.session_state.dataset

        # Display dataset stats
        num_rows = df.shape[0]
        num_cols = df.shape[1]
        num_numeric = df.select_dtypes(include=['number']).shape[1]
        num_categorical = df.select_dtypes(include=['object', 'category']).shape[1]

        st.markdown("---")
        st.markdown("**Dataset Summary:**")
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric(label="Records", value=num_rows)
        stat2.metric(label="Total Features", value=num_cols)
        stat3.metric(label="Numerical Features", value=num_numeric)
        stat4.metric(label="Categorical Features", value=num_categorical)

        st.markdown("---")
        st.write("**Dataset Preview:**")
        st.dataframe(st.session_state.dataset.head())

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.active_tab += 1
            st.rerun()

# -------------------------
# Step 2: Train Target Model
# -------------------------
if st.session_state.active_tab == 1:
    if st.session_state.dataset is not None:
        st.write("Preview of the dataset:")
        st.dataframe(st.session_state.dataset.head())

        default_index = 0
        if st.session_state.target_column in st.session_state.dataset.columns:
            default_index = st.session_state.dataset.columns.get_loc(st.session_state.target_column)

        numeric_columns = st.session_state.dataset.select_dtypes(include=['number']).columns
        if st.session_state.target_column in numeric_columns:
            default_numeric_index = list(numeric_columns).index(st.session_state.target_column)
        else:
            default_numeric_index = 0
        target_column = st.selectbox("Select target column (numerical only):", numeric_columns, index=default_numeric_index)

        st.write("Select target model type:")
        model_type = option_menu(
            menu_title="",
            options=["XGBoost", "Random Forest"],
            icons=["bar-chart-line", "tree"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f2f6", "justify-content": "center"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "5px",
                    "padding": "10px",
                    "--hover-color": "#E8F5E9",
                },
                "nav-link-selected": {
                    "background-color": "#4CAF50",
                    "color": "white",
                    "font-weight": "bold",
                    "border-radius": "6px",
                },
            }
        )

        st.session_state.target_model_type = model_type
        st.session_state.target_column = target_column

        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            if st.button("Train Target Model", use_container_width=True):
                if model_type == "XGBoost":
                    model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_target_model(st.session_state.dataset, target_column, model_type="xgb")
                else:
                    model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_target_model(st.session_state.dataset, target_column, model_type="rf")

                st.session_state.target_model = model
                st.session_state.target_X_train = X_train
                st.session_state.target_X_test = X_test
                st.session_state.target_X_ext = X_ext
                st.session_state.target_y_train = y_train
                st.session_state.target_y_test = y_test
                st.session_state.target_y_ext = y_ext
                st.session_state.target_r2 = r2_score
                st.session_state.target_of_ratio = of_ratio

                st.success(f"Target model trained successfully! R² score: {r2_score:.2f}, Overfit ratio: {of_ratio:.0f}")

    else:
        st.info("Please load a dataset in Step 1 first.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_tab -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.active_tab += 1
            st.rerun()

# -------------------------
# Step 3: Train Shadow Models
# -------------------------
if st.session_state.active_tab == 2:
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

            st.markdown("### Shadow Model Performance")
            perf_df = pd.DataFrame(stats, columns=["Overfit Ratio", "R² Score"], index=[f"Shadow Model {i+1}" for i in range(len(stats))])
                    # Add target model row at the top
            target_row = pd.DataFrame({
                "Overfit Ratio": [st.session_state.target_of_ratio],
                "R² Score": [st.session_state.target_r2]
            }, index=["**Target Model**"])

            perf_df = pd.concat([target_row, perf_df])
            st.table(perf_df.style.format({"Overfit Ratio": "{:.0f}", "R² Score": "{:.2f}"}))

            st.success("Shadow models trained and stored successfully!")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_tab -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.active_tab += 1
            st.rerun()


# -------------------------
# Step 4: Compute Features Sequentially
# -------------------------
if st.session_state.active_tab == 3:
    st.subheader("Step 4: Compute Attack Features")

    feature_sequence = ["Error", "Augmentation", "Missing", "Ensemble Variation", "Combining"]
    current = st.session_state.feature_stage
    current_label = feature_sequence[current]

    # Display feature progress
    completed = st.session_state.feature_completed
    substeps = ["Error", "Augmentation", "Missing", "Ensemble Variation"]
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
                    st.session_state.features_s_means_list = []

                    for split in st.session_state.shadow_splits:
                        X_train = split[0]  # X_train is the first element in the tuple
                        st.session_state.features_s_means = X_train.mean()
                        st.session_state.features_s_means_list.append(st.session_state.features_s_means)

                    progress.progress(100)
                    st.session_state.feature_completed[0] = True
                    st.session_state.feature_stage += 1
                    st.rerun()

    elif current_label == "Augmentation":

        label_col, select_col = st.columns([2, 4])
        with label_col:
            st.markdown("Selct noise levels:")
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
            
            if 1.0 in noise_levels:
                st.markdown("**Processing augmentation @ noise=1.0**")
                results = parallel_process_rows(st.session_state.target_X_train, feature_scale=1, model=st.session_state.target_model, 
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_train_t_1 = pd.DataFrame(results, columns=columns)

                results = parallel_process_rows(st.session_state.target_X_test, feature_scale=1, model=st.session_state.target_model,
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_test_t_1 = pd.DataFrame(results, columns=columns)

                st.session_state.augmented_results_1 = build_augmented_feature_dfs(st.session_state.shadow_models, st.session_state.shadow_splits, feature_scale=1, columns=columns, noise_label='1', augmented_records=augmentation_count)
            
            if 0.1 in noise_levels:
                st.markdown("**Processing augmentation @ noise=0.1**")
                results = parallel_process_rows(st.session_state.target_X_train, feature_scale=0.1, model=st.session_state.target_model, 
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_train_t_01 = pd.DataFrame(results, columns=columns)

                results = parallel_process_rows(st.session_state.target_X_test, feature_scale=0.1, model=st.session_state.target_model,
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_test_t_01 = pd.DataFrame(results, columns=columns)

                st.session_state.augmented_results_01 = build_augmented_feature_dfs(st.session_state.shadow_models, st.session_state.shadow_splits, feature_scale=0.1, columns=columns, noise_label='01', augmented_records=augmentation_count)
            
            if 0.01 in noise_levels:
                st.markdown("**Processing augmentation @ noise=0.01**")
                results = parallel_process_rows(st.session_state.target_X_train, feature_scale=0.01, model=st.session_state.target_model, 
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_train_t_001 = pd.DataFrame(results, columns=columns)

                results = parallel_process_rows(st.session_state.target_X_test, feature_scale=0.01, model=st.session_state.target_model,
                                                augmented_records=augmentation_count, desc="target model train set")
                st.session_state.aug_test_t_001 = pd.DataFrame(results, columns=columns)

                st.session_state.augmented_results_001 = build_augmented_feature_dfs(st.session_state.shadow_models, st.session_state.shadow_splits, feature_scale=0.01, columns=columns, noise_label='001', augmented_records=augmentation_count)
            
            #st.session_state.aug_results = results

            st.session_state.feature_completed[1] = True
            st.session_state.feature_stage += 1
            st.rerun()

    elif current_label == "Missing":        
        centered_col = st.columns([1, 3, 1])[1]
        with centered_col:
            strategy = st.radio("**Choose strategy:**", ["mean", "zero"], horizontal=True)
            n_important = st.slider("Choose number of missing features:", min_value=1, max_value=5, value=3)
            
            clicked = st.button("Compute Missing Value Features", use_container_width=True)
                
        if clicked:
            st.session_state.missing_train_t_stds, st.session_state.missing_train_t_entropies, st.session_state.missing_train_t_vars = calculate_missing_stats(st.session_state.target_X_train, 
                                                                                                            st.session_state.target_model, 
                                                                                                            st.session_state.feature_importance_t, 
                                                                                                            st.session_state.features_t_means, 
                                                                                                            strategy=strategy, 
                                                                                                            n_important=n_important,
                                                                                                            desc="target model train set")
            st.session_state.missing_test_t_stds, st.session_state.missing_test_t_entropies, st.session_state.missing_test_t_vars = calculate_missing_stats(st.session_state.target_X_test, 
                                                                                                            st.session_state.target_model, 
                                                                                                            st.session_state.feature_importance_t, 
                                                                                                            st.session_state.features_t_means, 
                                                                                                            strategy=strategy, 
                                                                                                            n_important=n_important,
                                                                                                            desc="target model test set")

            st.session_state.missing_train_stats = []
            st.session_state.missing_test_stats = []

            for model_idx, (model, split, feat_importance, feat_means) in enumerate(zip(st.session_state.shadow_models, st.session_state.shadow_splits,
                                                                                        st.session_state.model_importances, st.session_state.features_s_means_list)):
                X_train, X_test = split[0], split[1]

                # Train set
                train_stats = calculate_missing_stats(X_train, model, feat_importance, feat_means, strategy=strategy, n_important=n_important, desc=f"shadow model {model_idx+1} train set")
                st.session_state.missing_train_stats.append(train_stats)

                # Test set
                test_stats = calculate_missing_stats(X_test, model, feat_importance, feat_means, strategy=strategy, n_important=n_important, desc=f"shadow model {model_idx+1} test set")
                st.session_state.missing_test_stats.append(test_stats)

                st.session_state.feature_completed[2] = True
                st.session_state.feature_stage += 1
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

            for model_idx, (model, split) in enumerate(zip(st.session_state.shadow_models, st.session_state.shadow_splits)):
                X_train, X_test = split[0], split[1]

                # Train metrics
                train_metrics = est_func(model, X_train, desc=f"shadow model {model_idx+1} train set")
                st.session_state.ens_var_train_metrics.append(train_metrics)

                # Test metrics
                test_metrics = est_func(model, X_test, desc=f"shadow model {model_idx+1} test set")
                st.session_state.ens_var_test_metrics.append(test_metrics)
                
            st.session_state.feature_completed[3] = True
            st.success("All features computed successfully!")
            st.session_state.feature_stage += 1
            st.rerun()

    elif current_label == "Combining":
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
                        data_attack_dict[col] = list(df[col])

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
   
        st.session_state.data_attack_dict = data_attack_dict
        st.session_state.data_test_dict = data_test_dict
        st.success("All features combined successfully!")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_tab -= 1
            st.session_state.feature_stage = 0
            st.rerun()
    with col2:
        if current == len(feature_sequence) - 1:
            if st.button("Next ➡", use_container_width=True):
                st.session_state.active_tab += 1
                st.rerun()

# -------------------------
# Step 5: Train & Test Inference Model
# -------------------------
if st.session_state.active_tab == 4:
    st.subheader("Step 5: Train & Test Inference Model")

    # Group features by prefix
    from collections import defaultdict

    grouped = defaultdict(list)
    for key in st.session_state.data_attack_dict.keys():
        if key in ["index", "membership"]:
            continue
        if key.startswith("aug_pred_"):
            grouped["AUG_raw_preds"].append(key)
        elif key.startswith("aug_"):
            grouped["AUG_stats"].append(key)
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

        plot_metrics_three_sets(
            (y_train_att, pred_shadow_train, proba_shadow_train),
            (y_test_att, pred_shadow_test, proba_shadow_test),
            (yt_attack_test, pred_target, proba_target),
            labels=('Shadow train set', 'Shadow test set', 'Target')
        )

        def compute_metrics(y_true, y_pred, y_proba):
            return {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_proba),
                "TPR@FPR=0.1": compute_tpr_at_fpr(y_true, y_proba, target_fpr=0.1)
            }

        st.session_state.metrics_target = compute_metrics(yt_attack_test, pred_target, proba_target)

    # אתחול טבלת סיכום אם לא קיימת
    if "inference_results_table" not in st.session_state:
        st.session_state.inference_results_table = pd.DataFrame()

    # כפתור להוספת שורה
    if st.button("📥 Add Target Results to Summary Table", use_container_width=True):
        row = {}

        # ציון קבוצות הפיצ'רים שנבחרו (True/False)
        for group in feature_groups:
            row[f"{group}"] = group in selected_groups

        # הוספת המטריקות
        for metric_name, value in st.session_state.metrics_target.items():
            row[metric_name] = round(value, 2)

        # הוספת השורה לטבלה
        st.session_state.inference_results_table = pd.concat(
            [st.session_state.inference_results_table, pd.DataFrame([row])],
            ignore_index=True
        )
        st.success("✅ Target results added to summary table!")

    # הצגת הטבלה
    if not st.session_state.inference_results_table.empty:
        st.markdown("### 📊 Target Results Summary")
        st.dataframe(st.session_state.inference_results_table, use_container_width=True)


        #st.success("Inference model trained and applied to test data!")
        #st.markdown(f"**TPR@FPR=0.1:** {metrics.get('tpr@fpr=0.1', 0):.2f}")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅ Back", use_container_width=True):
            st.session_state.active_tab -= 1
            st.rerun()
