import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from xgboost import XGBRegressor
from dataset_app.utils.general import *

def train_RF_model(df, feature_name, ext_size=0.2, test_size=0.5, n_est=100, rand_stat=42, max_d=5, bootstrap_val=True, verbose=1):
    # Fill NaN values with the mean of each column
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df.fillna(numeric_df.mean(), inplace=True)

    for col in numeric_df.columns:
        df[col] = numeric_df[col]

    # Split the data to features and target variable
    X = df.drop(columns=[feature_name])
    y = df[feature_name]

    # Identify categorical features automatically
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Drop categorical variables
    X = X.drop(columns=cat_features)

    # Split data into original and external sets
    if ext_size > 0:
        X_orig, X_ext, y_orig, y_ext = train_test_split(X, y, test_size=ext_size, random_state=42)
    else:
        X_orig, X_ext, y_orig, y_ext = X, [], y, []

    # Split data into train and test sets
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=test_size, random_state=rand_stat)
    else:
        X_train, X_test, y_train, y_test = X, [], y, []

    # Train Regression model
    # model = RandomForestRegressor(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5, oob_score=True) experiment 1
    model = RandomForestRegressor(n_estimators=n_est, random_state=rand_stat, max_depth=max_d, min_samples_split=10, min_samples_leaf=5, oob_score=True, bootstrap=bootstrap_val)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    relative_train_rmse = train_rmse / np.mean(y_train)
    train_errors = y_train - train_preds

    if test_size > 0:
        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        relative_test_rmse = test_rmse / np.mean(y_test)
        test_errors = y_test - test_preds
        r2_score = model.score(X_test, y_test)
    else:
        test_preds = []
        test_rmse = np.nan
        relative_test_rmse = np.nan
        test_errors = []
        r2_score = 0

    overfit_percentage = ((test_rmse - train_rmse) / train_rmse) * 100  # Percentage increase

    # Find the minimum and maximum values, ignoring NaNs
    min_value = np.nanmin(np.concatenate([train_errors, test_errors]))
    max_value = np.nanmax(np.concatenate([train_errors, test_errors]))

    # Set the bin edges
    bin_edges = np.linspace(min_value, max_value, 50)

    # Print metrics and plot prediction errors
    if verbose == 1:
        print(f"R2 Score = {r2_score:.4f} | Overfit_percentage = {overfit_percentage:.4f}")
        #print(f"Train set: RMSE = {train_rmse:.4f}, Relative RMSE = {relative_train_rmse:.4f}")
        #print(f"Test set: RMSE = {test_rmse:.4f}, Relative RMSE = {relative_test_rmse:.4f}")
        #print(f"Test R2 score = {r2_score:.4f}")
        #print(f"Overfit_percentage = {overfit_percentage:.4f}")
        #plt.hist(train_errors, bins=bin_edges, alpha=0.5, label='Train')
        #plt.hist(test_errors, bins=bin_edges, alpha=0.5, label='Test')
        #plt.legend()
        #plt.xlabel('Prediction error')
        #plt.ylabel('Frequency')
        #plt.show()

    return model, X_train, X_test, X_ext ,y_train, y_test, y_ext, overfit_percentage, r2_score

def train_XGB_model(df, feature_name, ext_size=0.2, test_size=0.5, rand_stat=42, n_est=100, max_d=5, verbose=1):
    # Split the data to features and target variable
    X = df.drop(columns=[feature_name])
    y = df[feature_name]

    # Identify categorical features automatically
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    # Drop categorical variables
    X = X.drop(columns=cat_features)

    # Fill NaN values with the mean of each column
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    # Split data into original and external sets
    if ext_size > 0:
        X_orig, X_ext, y_orig, y_ext = train_test_split(X, y, test_size=ext_size, random_state=42)
    else:
        X_orig, X_ext, y_orig, y_ext = X, [], y, []

    # Split data into train and test sets
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=test_size, random_state=rand_stat)
    else:
        X_train, X_test, y_train, y_test = X, [], y, []

    # Train Regression model
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = max_d, alpha = 10, n_estimators = n_est, n_jobs=-1, random_state=rand_stat)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    train_preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    relative_train_rmse = train_rmse / np.mean(y_train)
    train_errors = y_train - train_preds

    if test_size > 0:
        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        relative_test_rmse = test_rmse / np.mean(y_test)
        test_errors = y_test - test_preds
        r2_score = model.score(X_test, y_test)
    else:
        test_preds = []
        test_rmse = np.nan
        relative_test_rmse = np.nan
        test_errors = []
        r2_score = 0

    overfit_percentage = ((test_rmse - train_rmse) / train_rmse) * 100  # Percentage increase
    
    # Find the minimum and maximum values, ignoring NaNs
    min_value = np.nanmin(np.concatenate([train_errors, test_errors]))
    max_value = np.nanmax(np.concatenate([train_errors, test_errors]))

    # Set the bin edges
    bin_edges = np.linspace(min_value, max_value, 50)

    # Print metrics and plot prediction errors
    if verbose == 1:
        #print(f"Train set: RMSE = {train_rmse:.4f}, Relative RMSE = {relative_train_rmse:.4f}")
        #print(f"Test set: RMSE = {test_rmse:.4f}, Relative RMSE = {relative_test_rmse:.4f}")
        print(f"R2 Score = {r2_score:.4f} | Overfit_percentage = {overfit_percentage:.4f}")
        #plt.hist(train_errors, bins=bin_edges, alpha=0.5, label='Train')
        #plt.hist(test_errors, bins=bin_edges, alpha=0.5, label='Test')
        #plt.legend()
        #plt.xlabel('Prediction error')
        #plt.ylabel('Frequency')
        #plt.show()

    return model, X_train, X_test, X_ext ,y_train, y_test, y_ext, overfit_percentage, r2_score

def train_target_model(df, target_column, model_type="rf"):
    # if model_type == "CatBoost":
    #     return train_cat_model(df, target_column)
    # elif model_type == "Linear":
    #     return train_linreg_model(df, target_column)
    if model_type == "xgb":
        return train_XGB_model(df, target_column)
    # elif model_type == "DecisionTree":
    #     return train_DT_model(df, target_column)
    elif model_type == "rf":
        return train_RF_model(df, target_column)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

def train_shadow_models(train_function, data, target, param_dicts, ext_size=0.2):
    """
    Trains shadow models using the provided train_function with different parameters.
    """
    models = []
    data_splits = []
    model_stats = []

    for params in stqdm(param_dicts, desc="Training shadow models"):
        
        model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_function(
            data, target,
            n_est=params.get('Estimators', 50),
            rand_stat = params.get('Seed', 0),
            max_d=params.get('Max Depth', 4),
            test_size=params.get('Test Size', 0.5),
            ext_size=ext_size
        )
        models.append(model)
        data_splits.append((X_train, X_test, X_ext, y_train, y_test, y_ext))
        model_stats.append((of_ratio, r2_score))

    return models, data_splits, model_stats

def generate_target_models(train_function, dataset, target_col, owned_data_ratio=0.5,
                           holdout_ratio=0.0, param_dicts=None, random_state=42):
    """
    Generates multiple target models for dataset-level MIA.

    Args:
        train_function: Function that returns model, splits, and stats.
        dataset: Full dataset including the target column.
        target_col: Name of the target column.
        owned_model_ratio: Ratio of models trained with the owned data.
        owned_data_ratio: Ratio of owned data from all the data.
        holdout_ratio: Ratio of holdout data from all the data.
        param_dicts: List of parameter dictionaries for each model.
        random_state: Seed for reproducibility.

    Returns:
        models, data_splits, model_stats, (owned_X, owned_y), (external_X, external_y)
    """

    # Shuffle the full dataset
    dataset = shuffle(dataset, random_state=random_state)

    models = []
    data_splits = []
    model_stats = []

    # שלב 1: פיצול holdout מה-dataset המקורי
    holdout_df = dataset.sample(frac=holdout_ratio, random_state=42)
    working_df = dataset.drop(holdout_df.index).reset_index(drop=True)

    # שלב 2: המשך חישוב על working_df
    n_owned = int(len(working_df) * owned_data_ratio)

    owned_df = working_df.iloc[:n_owned]
    external_df = working_df.iloc[n_owned:]

    train_size = min(len(owned_df), len(external_df))
   
    for i, params in stqdm(enumerate(param_dicts), total=len(param_dicts), desc="Training target models"):
       
        # שלב 3: בניית סט אימון לפי ratio
        own_ex_ratio = params.get('Owned to External Ratio', 0.5)

        # max_owned = len(owned_df)
        # max_external = len(external_df)

        # max_total_size_owned_limited = max_owned / own_ex_ratio if own_ex_ratio > 0 else float('inf')
        # max_total_size_external_limited = max_external / (1 - own_ex_ratio) if own_ex_ratio < 1 else float('inf')

        # max_total_size = int(min(max_total_size_owned_limited, max_total_size_external_limited))

        # n_owned = int(max_total_size * own_ex_ratio)
        # n_external = max_total_size - n_owned

        n_owned = int(train_size * own_ex_ratio)
        n_external = train_size - n_owned

        sampled_owned = owned_df.sample(n=n_owned, random_state=42)
        sampled_external = external_df.sample(n=n_external, random_state=42)

        train_df = pd.concat([sampled_owned, sampled_external])
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Training model {i+1}/{len(param_dicts)} with owned:external ratio = {own_ex_ratio:.2f} | n_total = {n_owned+n_external}, n_owned = {n_owned}, n_external = {n_external}")

        # Train the model
        model, X_train, X_test, X_ext, y_train, y_test, y_ext, of_ratio, r2_score = train_function(
            train_df,
            target_col,
            n_est=params.get('Estimators', 100),
            rand_stat=params.get('Seed', random_state + i),
            max_d=params.get('Max Depth', 6),
            test_size=0.5,
            ext_size=0.0
        )

        models.append(model)
        data_splits.append((X_train, X_test, X_ext, y_train, y_test, y_ext))
        model_stats.append((of_ratio, r2_score, own_ex_ratio))

    return models, data_splits, model_stats, owned_df, external_df