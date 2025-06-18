import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from app.utils.general import *

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
        print(f"Train set: RMSE = {train_rmse:.4f}, Relative RMSE = {relative_train_rmse:.4f}")
        print(f"Test set: RMSE = {test_rmse:.4f}, Relative RMSE = {relative_test_rmse:.4f}")
        print(f"Test R2 score = {r2_score:.4f}")
        print(f"Overfit_percentage = {overfit_percentage:.4f}")
        plt.hist(train_errors, bins=bin_edges, alpha=0.5, label='Train')
        plt.hist(test_errors, bins=bin_edges, alpha=0.5, label='Test')
        plt.legend()
        plt.xlabel('Prediction error')
        plt.ylabel('Frequency')
        plt.show()

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
        print(f"Train set: RMSE = {train_rmse:.4f}, Relative RMSE = {relative_train_rmse:.4f}")
        print(f"Test set: RMSE = {test_rmse:.4f}, Relative RMSE = {relative_test_rmse:.4f}")
        print(f"R2 Score = {r2_score:.4f}")
        print(f"Overfit_percentage = {overfit_percentage:.4f}")
        plt.hist(train_errors, bins=bin_edges, alpha=0.5, label='Train')
        plt.hist(test_errors, bins=bin_edges, alpha=0.5, label='Test')
        plt.legend()
        plt.xlabel('Prediction error')
        plt.ylabel('Frequency')
        plt.show()

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
            n_est=params.get('n_est', 50),
            rand_stat=params.get('rand_stat', 0),
            max_d=params.get('max_d', 4),
            ext_size=ext_size
        )
        models.append(model)
        data_splits.append((X_train, X_test, X_ext, y_train, y_test, y_ext))
        model_stats.append((of_ratio, r2_score))

    return models, data_splits, model_stats