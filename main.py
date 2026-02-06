# %%

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
bolig_data = pd.read_csv("data/boligsalg.csv", sep=";")
bolig_data = gpd.GeoDataFrame(
    bolig_data,
    geometry=gpd.points_from_xy(bolig_data.Longitude, bolig_data.Latitude),
    crs="EPSG:4326",
)

turbine_data = pd.read_excel("data/Vindmølledata til 2025-01.xlsx", skiprows=10)


# %%


turbine_data.info()

# %%
