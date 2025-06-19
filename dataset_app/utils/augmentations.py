# app/utils/augmentations.py

import numpy as np
import pandas as pd
from scipy.stats import entropy
import streamlit as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import streamlit as st
import time
from dataset_app.utils.general import *

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



def rename_augmented_columns(df, noise_label):
    renamed = {}
    for col in df.columns:
        if col in ['aug_preds_diff', 'aug_preds_var', 'aug_preds_range']:
            renamed[col] = f'{col}_{noise_label}'
        elif col.startswith('aug_pred_'):
            pred_id = col.replace('aug_pred_', '')
            renamed[col] = f'aug_pred_{noise_label}_{pred_id}'
    return df.rename(columns=renamed)


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
            augmented_df = pd.DataFrame(augmented_array, columns=row.index)

            preds = model.predict(augmented_df)
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
    #st.toast(f"✔ Processing completed in {duration:.2f} seconds")
    print(f"[⏱] parallel_process_rows (flexible) took {duration:.2f} seconds")
    
    return results
#--------------------------

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