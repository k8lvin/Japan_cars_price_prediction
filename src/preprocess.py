import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

TARGET = "price_usd"

engine = create_engine("postgresql+psycopg2://postgres:1234@localhost:5432/car_import")


def load_data():
    df = pd.read_sql("SELECT * FROM cardata.carjapan", engine)
    
    return df


def clean_data(df):
    df[TARGET] = pd.to_numeric (df[TARGET].str.replace(",", ""), errors="coerce")
    #converts price_usd to numeric and removes delimeter(,)
    
    df["options"] = df["options"].apply(lambda x: 0
    if pd.isna(x) or str(x).strip().lower() == "nan" or str(x).strip() == ""
    else len([opt.strip() for opt in str(x).split("|") if opt.strip()])) 
    # checks for missing values NaN "nan" or empty strings and replace with 0, separates texts where delimeter is (|)
    
    df = df.dropna(subset=[TARGET]) 
    # removing NaN rows from the target (price_usd)

    df[TARGET] = np.log1p(df[TARGET])

    return df


def define_columns():
    num_cols = ["year", "mileage_km", "engine_cc", "has_sunroof","options","has_leather",
            "has_navigation", "has_alloy_wheels", "has_4wd", "has_airbag", "has_abs", "has_camera"]

    onehot_cols = ["fuel","transmission","vehicle_type"]

    target_cols = ["name","chassis_code"]
    
    return num_cols, onehot_cols, target_cols


def build_preprocessor(num_cols: list, onehot_cols: list, target_cols: list) -> ColumnTransformer:
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore',drop='first'))
    ])

    target_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("onehot", onehot_pipeline, onehot_cols),
        ("target", target_pipeline, target_cols)
    ])

    return preprocessor

