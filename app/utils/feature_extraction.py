import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import entropy
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import streamlit as st
import time
from app.utils.general import *

def _init_globals(target_model, shadow_models, shadow_splits):
    set_global_target_model(target_model)
    set_global_shadow_models(shadow_models)
    set_global_shadow_splits(shadow_splits)

# Global model and split lists for multiprocessing
_global_target_model = None
_global_shadow_models = []
_global_shadow_splits = []
_global_feature_means_list = []
_global_feature_medians_list = []
_global_feature_importances = []

def set_global_target_model(model):
    global _global_target_model
    _global_target_model = model

def get_global_target_model():
    return _global_target_model

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
    return _global_feature_importances[model_id]

def rename_augmented_columns(df, noise_label):
    renamed = {}
    for col in df.columns:
        if col in ['aug_preds_diff', 'aug_preds_var', 'aug_preds_range']:
            renamed[col] = f'{col}_{noise_label}'
        elif col.startswith('aug_pred_'):
            pred_id = col.replace('aug_pred_', '')
            renamed[col] = f'aug_pred_{noise_label}_{pred_id}'
    return df.rename(columns=renamed)

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

def process_rows(X, feature_scale, model, augmented_records, desc):

    results = []
    for args in stqdm([(index, row, feature_scale, model, augmented_records) for index, row in X.iterrows()], total=len(X), desc=desc):
      results.append(process_row(args))

    return results

def parallel_process_rows_old(X, feature_scale, model_id, augmented_records, desc, max_workers=4):
    args_list = [(index, row, feature_scale, model_id, augmented_records) for index, row in X.iterrows()]
    total = len(args_list)
    results = [None] * total
    start_time = time.time()

    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(f"{desc} ({total} records):")
    with progress_col:
        progress = st.progress(0)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_globals, initargs=(_global_target_model, _global_shadow_models, _global_shadow_splits)) as executor:
        futures = {executor.submit(process_row, args): i for i, args in enumerate(args_list)}

        for completed_idx, future in enumerate(as_completed(futures)):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"⚠️ Error processing row {i}: {e}")
                results[i] = (i, 0.0, 0.0, 0.0, *[0.0] * augmented_records)
            progress.progress(int((completed_idx + 1) / total * 100))

    duration = time.time() - start_time
    st.toast(f"✔ Processing completed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_rows took {duration:.2f} seconds")
    return results

def parallel_process_rows(X, feature_scale, model_id, augmented_records, desc, max_workers=4, batch_size=16):
    args_list = [(index, row, feature_scale, model_id, augmented_records) for index, row in X.iterrows()]
    batches = [args_list[i:i+batch_size] for i in range(0, len(args_list), batch_size)]

    results = []
    start_time = time.time()
    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(f"{desc} ({len(args_list)} records):")
    with progress_col:
        progress = st.progress(0)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_globals, initargs=(_global_target_model, _global_shadow_models, _global_shadow_splits)) as executor:
        futures = {executor.submit(process_row_batch, batch): i for i, batch in enumerate(batches)}
        total_batches = len(batches)

        for completed_idx, future in enumerate(as_completed(futures)):
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                print(f"⚠️ Error processing batch {completed_idx}: {e}")
            progress.progress(int((completed_idx + 1) / total_batches * 100))

    duration = time.time() - start_time
    st.toast(f"✔ Processing completed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_rows (batched) took {duration:.2f} seconds")
    return results

# def build_augmented_feature_dfs_old(feature_scale, columns, noise_label, augmented_records, max_workers=4):
#     all_results = []

#     for model_idx in range(len(_global_shadow_splits)):
#         X_train, X_test, *_ = get_global_shadow_split(model_idx)

#         # Process train set
#         train_results = parallel_process_rows(
#             X_train,
#             feature_scale=feature_scale,
#             model_id=model_idx + 1,
#             augmented_records=augmented_records,
#             desc=f"Shadow model {model_idx+1} train set",
#             max_workers=max_workers
#         )
#         train_df = pd.DataFrame(train_results, columns=columns)
#         train_df = rename_augmented_columns(train_df, noise_label)
#         train_df['model_id'] = model_idx
#         train_df['split'] = 'train'
#         all_results.append(train_df)

#         # Process test set
#         test_results = parallel_process_rows(
#             X_test,
#             feature_scale=feature_scale,
#             model_id=model_idx + 1,
#             augmented_records=augmented_records,
#             desc=f"Shadow model {model_idx+1} test set",
#             max_workers=max_workers
#         )
#         test_df = pd.DataFrame(test_results, columns=columns)
#         test_df = rename_augmented_columns(test_df, noise_label)
#         test_df['model_id'] = model_idx
#         test_df['split'] = 'test'
#         all_results.append(test_df)

#     return pd.concat(all_results, ignore_index=True)

def build_augmented_feature_dfs(shadow_models, shadow_splits, feature_scale, columns, noise_label,
                                 augmented_records, max_workers=4, batch_size=16, use_joblib=False):
    all_results = []

    for model_idx, (model, split) in enumerate(zip(shadow_models, shadow_splits)):
        X_train, X_test = split[0], split[1]

        train_results = parallel_process_rows_flexible(
            X=X_train,
            feature_scale=feature_scale,
            model_id=model_idx + 1,
            model=model,
            augmented_records=augmented_records,
            desc=f"Shadow model {model_idx+1} train set",
            max_workers=max_workers,
            batch_size=batch_size,
            use_joblib=use_joblib
        )

        train_df = pd.DataFrame(train_results, columns=columns)
        train_df = rename_augmented_columns(train_df, noise_label)
        train_df['model_id'] = model_idx
        train_df['split'] = 'train'
        all_results.append(train_df)

        test_results = parallel_process_rows_flexible(
            X=X_test,
            feature_scale=feature_scale,
            model_id=model_idx + 1,
            model=model,
            augmented_records=augmented_records,
            desc=f"Shadow model {model_idx+1} test set",
            max_workers=max_workers,
            batch_size=batch_size,
            use_joblib=use_joblib
        )

        test_df = pd.DataFrame(test_results, columns=columns)
        test_df = rename_augmented_columns(test_df, noise_label)
        test_df['model_id'] = model_idx
        test_df['split'] = 'test'
        all_results.append(test_df)

    return pd.concat(all_results, ignore_index=True)


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

def compute_estimators_metrics(model, X_data, alpha=0.01, window_prcnt=0.1, desc=""):
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

    for i in stqdm(range(len(X_data)), desc=desc):
        dmatrix = xgb.DMatrix(X_data.iloc[[i]] if isinstance(X_data, pd.DataFrame) else X_data[i:i+1])

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

    return directional_consistencies, convergence_indexs

def calculate_tree_stats(model, X, desc):
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

# Create a test set for a row of real data with guassian noise
def create_augmented_set(row, n_cases=3, feature_scale=0.2):
    augmented_set = [row]

    # Make copies of row
    for _ in range(n_cases):
        # Standardize the original row and use the calculated std for the normal distribution
        #scaler = StandardScaler().fit(np.array([row]))
        #standardized_row = scaler.transform(np.array([row]))[0]
        
        # Create vector of random gaussians using the standardized std
        noise = np.random.normal(loc=0.0, scale=feature_scale*np.abs(row), size=len(row))
        #noise = np.random.uniform(low=-feature_scale*np.abs(row), high=feature_scale*np.abs(row), size=len(row))
        
        # Add to the standardized row to get the new row
        new_row = row + noise
        
        # Store in test set
        augmented_set.append(new_row)

    return augmented_set

def process_row(args):
    index, row, feature_scale, model_id, n_cases = args
    #start_time = time.time()
    model = get_global_shadow_model(model_id - 1) if model_id > 0 else get_global_target_model()
    augmented_set = create_augmented_set(list(row), n_cases=n_cases, feature_scale=feature_scale)
    augmented_set_df = pd.DataFrame(augmented_set, columns=row.index)
    preds = model.predict(augmented_set_df)
    orig_pred = preds[0]
    aug_preds = preds[1:]
    pred_range = np.max(aug_preds) - np.min(aug_preds)
    diff = abs(orig_pred - np.mean(aug_preds))
    pred_features = list(aug_preds)
    #duration = time.time() - start_time
    #print(f"[🧠] Row {index} processed by PID {os.getpid()} in {duration:.2f} sec")
    return (index, np.var(aug_preds), pred_range, diff, *pred_features)

def process_row_batch(args_batch):
    """
    Batch version of process_row. Each item in args_batch is a tuple of:
    (index, row, feature_scale, model_id, n_cases)
    """
    # נניח שכולם שייכים לאותו מודל
    model_id = args_batch[0][3]
    model = get_global_shadow_model(model_id - 1) if model_id > 0 else get_global_target_model()

    batch_results = []
    for index, row, feature_scale, _, n_cases in args_batch:
        augmented_set = create_augmented_set(row.values, n_cases=n_cases, feature_scale=feature_scale)
        augmented_set_df = pd.DataFrame(augmented_set, columns=row.index)
        preds = model.predict(augmented_set_df)
        orig_pred = preds[0]
        aug_preds = preds[1:]
        pred_range = np.max(aug_preds) - np.min(aug_preds)
        diff = abs(orig_pred - np.mean(aug_preds))
        batch_results.append((index, np.var(aug_preds), pred_range, diff, *aug_preds))
    return batch_results

#--------------------------
# Joblib version of process_row_batch
def create_args_list(X, feature_scale, model_id=None, model=None, n_cases=3, use_direct_model=False):
    if use_direct_model:
        return [(i, row, feature_scale, model, n_cases) for i, row in X.iterrows()]
    else:
        return [(i, row, feature_scale, model_id, n_cases) for i, row in X.iterrows()]

def process_row_batch_joblib(args_batch):
    """
    גרסה ל-joblib: מקבלת אצווה של שורות, כל אחת עם המודל עצמו.
    כל args: (index, row, feature_scale, model, n_cases)
    """
    results = []

    for index, row, feature_scale, model, n_cases in args_batch:
        try:
            row_values = row.values if hasattr(row, 'values') else np.array(row)
            augmented_rows = [row_values]
            for _ in range(n_cases):
                noise = np.random.normal(loc=0.0, scale=feature_scale * np.abs(row_values), size=row_values.shape)
                augmented_rows.append(row_values + noise)

            augmented_array = np.stack(augmented_rows)

            preds = model.predict(augmented_array)
            orig_pred = preds[0]
            aug_preds = preds[1:]

            var = np.var(aug_preds)
            pred_range = np.max(aug_preds) - np.min(aug_preds)
            diff = abs(orig_pred - np.mean(aug_preds))

            results.append((index, var, pred_range, diff, *aug_preds))

        except Exception as e:
            print(f"❗ Error in joblib batch row {index}: {e}")
            results.append((index, 0.0, 0.0, 0.0, *[0.0] * n_cases))

    return results

def parallel_process_rows_flexible(X, feature_scale, model_id=None, model=None, augmented_records=3,
                                   desc="", max_workers=4, batch_size=16, use_joblib=False):
    """
    גמיש: רץ או עם ProcessPoolExecutor או עם joblib בהתאם ל־use_joblib.
    אם use_joblib=True — נדרש להעביר את המודל המלא, לא רק model_id.
    """
    start_time = time.time()

    args_list = create_args_list(
        X=X,
        feature_scale=feature_scale,
        model_id=model_id,
        model=model,
        n_cases=augmented_records,
        use_direct_model=use_joblib
    )
    batches = [args_list[i:i+batch_size] for i in range(0, len(args_list), batch_size)]
    results = []

    desc_col, progress_col = st.columns([2, 3])
    with desc_col:
        st.markdown(f"{desc} ({len(args_list)} records):")
    with progress_col:
        progress = st.progress(0)

    if use_joblib:
        for completed_idx, batch_result in enumerate(
            Parallel(n_jobs=max_workers, backend="loky")(
                delayed(process_row_batch_joblib)(batch) for batch in batches
            )
        ):
            results.extend(batch_result)
            progress.progress(int((completed_idx + 1) / len(batches) * 100))

    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_globals,
                                 initargs=(_global_target_model, _global_shadow_models, _global_shadow_splits)) as executor:
            futures = {
                executor.submit(process_row_batch, batch): i
                for i, batch in enumerate(batches)
            }
            for completed_idx, future in enumerate(as_completed(futures)):
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"⚠️ Error in batch {completed_idx}: {e}")
                progress.progress(int((completed_idx + 1) / len(batches) * 100))
    
    duration = time.time() - start_time
    st.toast(f"✔ Processing completed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_rows (flexible) took {duration:.2f} seconds")
    
    return results
#--------------------------

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
    st.toast(f"✔ Missing stats processed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_missing_rows took {duration:.2f} seconds")
    return results
