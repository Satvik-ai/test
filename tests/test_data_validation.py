import pandas as pd
import pytest

def test_parquet_data_integrity():
    # Load Parquet data
    df = pd.read_parquet("data/iris.parquet")

    # 1. Check shape — it should have 7 columns
    assert df.shape[1] == 7, f"Expected 7 columns, found {df.shape[1]}"

    # 2. Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

    # 3. Check expected columns
    expected_cols = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "event_timestamp",
        "iris_v1_id",
    }
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

    # 4. Verify event_timestamp is datetime type
    assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]), \
        "'event_timestamp' column should be datetime type"

    # 5. Check ID column uniqueness
    assert df["iris_v1_id"].is_unique, "'iris_v1_id' column contains duplicate values"
    
def test_numeric_values_positive_parquet():
    # Load dataset from Parquet file
    df = pd.read_parquet("data/iris.parquet")

    # Define numeric columns
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Check all numeric values are positive (> 0)
    for col in numeric_cols:
        assert (df[col] > 0).all(), f"Negative or zero values found in column: {col}"