# app/utils/data_loading.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Dynamically resolve data directory based on current working directory
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "data"))

def load_house_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "house.csv"))
    df.drop("Id", axis=1, inplace=True)
    return df

def load_census_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "acs2015_census_tract_data.csv"))
    drop_columns = ["IncomeErr", "IncomePerCapErr", "Poverty", "ChildPoverty", "Professional", "Service"]
    df.drop(drop_columns, axis=1, inplace=True)
    df = df.sample(n=10000, random_state=42)  # Ensure reproducibility
    return df

def load_abalone_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "abalone.csv"))
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    return df

def load_kc_house_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "kc_house_data.csv"))
    df = df.drop(columns=["id", "date"])
    return df

def load_diamonds_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "diamonds.csv"))
    df.drop(["index"], axis=1, inplace=True)
    return df
