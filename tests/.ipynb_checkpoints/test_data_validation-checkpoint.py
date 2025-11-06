import pandas as pd
import pytest
import re

def test_parquet_data_integrity():
    # Load Parquet data
    df = pd.read_parquet("data/iris.parquet")

    # 1. Check shape — should have 7 columns
    assert df.shape[1] == 7, f"Expected 7 columns, found {df.shape[1]}"

    # 2. Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"

    # 3. Expected base columns (excluding versioned ID)
    expected_base_cols = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "event_timestamp",
    }
    # Find the versioned ID column dynamically
    versioned_id_col = next(
        (col for col in df.columns if re.match(r"iris_v\d+_id", col)),
        None
    )

    assert versioned_id_col is not None, \
        "No versioned ID column found (expected something like 'iris_v1_id' or 'iris_v2_id')"

    # 4. Verify all expected columns are present
    expected_cols = expected_base_cols.union({versioned_id_col})
    assert expected_cols.issubset(df.columns), \
        f"Missing columns: {expected_cols - set(df.columns)}"

    # 5. Verify event_timestamp is datetime type
    assert pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]), \
        "'event_timestamp' column should be datetime type"

    # 6. Check versioned ID column uniqueness
    assert df[versioned_id_col].is_unique, \
        f"'{versioned_id_col}' column contains duplicate values"

    
def test_numeric_columns_are_numeric_parquet():
    df = pd.read_parquet("data/iris.parquet")
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"