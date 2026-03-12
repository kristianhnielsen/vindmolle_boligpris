"""Diagnostic script to inspect the data structure."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))
from data_handler import DataHandler

dh = DataHandler(radius=2000)
data = dh.data_next.copy()

print("Data shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst few rows:")
print(data.head())

print("\nData types:")
print(data.dtypes)

print("\nChecking turbine-related columns:")
for col in ["has_new_turbine", "dist_to_new_turbine", "date_of_effect", "tilslutning_dato"]:
    if col in data.columns:
        print(f"  {col}: {data[col].dtype}, unique values: {data[col].nunique()}, nulls: {data[col].isna().sum()}")
        if col in ["date_of_effect", "tilslutning_dato"]:
            print(f"    Sample values: {data[col].dropna().head().tolist()}")

print("\nChecking date columns:")
for col in ["salgs_dato", "salgs_dato_prev"]:
    if col in data.columns:
        print(f"  {col}: {data[col].dtype}, nulls: {data[col].isna().sum()}")
        print(f"    Sample values: {data[col].dropna().head().tolist()}")

print("\nTreatment group stats:")
treatment = data[data["has_new_turbine"] == 1]
print(f"  Has new turbine: {(data['has_new_turbine'] == 1).sum()} rows")
print(f"  Distance stats:\n{data['dist_to_new_turbine'].describe()}")
