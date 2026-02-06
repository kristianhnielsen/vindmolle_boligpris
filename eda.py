import marimo

__generated_with = "0.19.4"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    # Overall Plan

    ## Feature Engineering

    - identify the houses with more than one sale
    - identify which turbines were active or planned to be active within the next 12 months
    - calculate the distance to the nearest turbine
    - get the area median house price at time of sale

    ## Regression
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    # Import libraries, load and clean data & join data
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from shapely import wkt
    import altair as alt
    return alt, gpd, mo, np, pd, plt, sns, wkt


@app.cell
def _(gpd, pd, wkt):
    _bolig_data_df = pd.read_csv("data/boligsalg.csv", sep=";")
    _bolig_data_df["Aar"] = pd.to_datetime(_bolig_data_df["Aar"], format="%Y")
    # _bolig_data_df['byg026OpFoerelsesAar'] = pd.to_datetime(_bolig_data_df['byg026OpFoerelsesAar'], format='%Y')
    _geometry_column = "byg404Koordinat"
    _bolig_data_df[_geometry_column] = _bolig_data_df[_geometry_column].apply(
        wkt.loads
    )

    bolig_data = gpd.GeoDataFrame(
        _bolig_data_df,
        geometry=_geometry_column,
        crs="EPSG:25832",
    )
    bolig_data.rename_geometry("geometry", inplace=True)
    bolig_data.drop(
        bolig_data[bolig_data["Valutakode"] == "EUR"].index, inplace=True
    )
    return (bolig_data,)


@app.cell
def _(pd):
    turbine_dk = pd.read_excel("data/Vindmølledata til 2025-01.xlsx", skiprows=10)
    return (turbine_dk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Turbine data
    - Can we make a multiplier for an area with multiple turbines within a short distance?
    - Find the centroid for that turbine park
    - Use the centroid as point of distance calculation
    """)
    return


@app.cell
def _(turbine_dk):
    turbine_dk.columns = turbine_dk.columns.astype(str)
    turbine_dk.rename(
        columns={
            "X (øst) koordinat \nUTM 32 Euref89": "x",
            "Y (nord) koordinat \nUTM 32 Euref89": "y",
            "Møllenummer (GSRN)": "id",
        },
        inplace=True,
    )
    turbine_vejle = turbine_dk[turbine_dk["Kommune"].str.contains("630")]
    return (turbine_vejle,)


@app.cell
def _(gpd, turbine_vejle):
    turbine_vejle_geo = gpd.GeoDataFrame(
        turbine_vejle[
            [
                "id",
                "x",
                "y",
                "Dato for oprindelig nettilslutning",
                "Dato for afmeldning",
                "Koordinatoprindelse",
            ]
        ],
        geometry=gpd.points_from_xy(
            x=turbine_vejle["x"], y=turbine_vejle["y"], crs="EPSG:25832"
        ),
    )
    turbine_vejle_geo.plot()
    return (turbine_vejle_geo,)


@app.cell
def _(turbine_vejle_geo):
    turbine_vejle_geo.dropna(inplace=True)
    turbine_vejle_geo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Each house's distance to turbines
    A dataframe with each house with at least 1 registered sale:
    - a list of turbines that were erect (valid) at the date where the sale happened
    - distance to every valid turbine
    - distance and ID of nearest valid turbine
    """)
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    # Turbine Clustering
    """)
    return


@app.cell(hide_code=True)
def _():
    from sklearn.cluster import KMeans
    return (KMeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    K = 28 seems like a good fit
    """)
    return


@app.cell(hide_code=True)
def _(turbine_vejle_geo):
    len(turbine_vejle_geo)
    return


@app.cell(hide_code=True)
def _(mo, turbine_vejle_geo):
    k_clusters_slider = mo.ui.slider(
        1, turbine_vejle_geo.shape[0], show_value=True
    )
    k_clusters_slider
    return (k_clusters_slider,)


@app.cell(hide_code=True)
def _(KMeans, alt, k_clusters_slider, turbine_vejle_geo):
    kmeans = KMeans(n_clusters=k_clusters_slider.value, random_state=42)
    kmeans_clustering = kmeans.fit(turbine_vejle_geo[["x", "y"]])
    turbine_vejle_geo["cluster_kmeans"] = kmeans_clustering.labels_
    # turbine_vejle_geo['cluster_center_x'] = kmeans_clustering.cluster_centers_[:,0]
    # turbine_vejle_geo['cluster_center_y'] = kmeans_clustering.cluster_centers_[:,1]

    alt.Chart(turbine_vejle_geo, title="K-Means").mark_circle(size=60).encode(
        x=alt.X("x", scale=alt.Scale(zero=False)),  # Tighten X axis
        y=alt.Y("y", scale=alt.Scale(zero=False)),  # Tighten Y axis
        color="cluster_kmeans:N",  # cluster column as categorical
        tooltip=["cluster_kmeans:N"],
    ).interactive().show()
    # alt.Chart(turbine_vejle_geo, title='K-Means').mark_circle(size=60).encode(
    #    x=alt.X('cluster_center_x', scale=alt.Scale(zero=False)),  # Tighten X axis
    #     y=alt.Y('cluster_center_y', scale=alt.Scale(zero=False)),  # Tighten Y axis
    #     color='cluster_kmeans:N',   # cluster column as categorical
    #     tooltip=['cluster_kmeans:N'],
    # ).interactive().show()
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Distance to Turbine
    """)
    return


@app.cell
def _(turbine_vejle_geo):
    turbine_vejle_geo
    return


@app.cell
def _(bolig_data, gpd, turbine_vejle_geo):
    result = gpd.sjoin_nearest(
        bolig_data, turbine_vejle_geo, how="left", distance_col="distance_m"
    )
    result
    return


@app.cell
def _():
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    # Houses with more than one sale
    """)
    return


@app.cell
def _(bolig_data):
    bolig_data["BFEnummer"].value_counts()
    return


@app.cell
def _(bolig_data):
    _bfe_count = bolig_data["BFEnummer"].value_counts()
    _bfe_more_than_two = _bfe_count[_bfe_count > 1].index
    bolig_data_multiple_sales = bolig_data[
        bolig_data["BFEnummer"].isin(_bfe_more_than_two)
    ].copy()

    print(f"Rows of all Vejle sales:\t{len(bolig_data)}")
    print(f"Rows with multiple sales:\t{len(bolig_data_multiple_sales)}")
    print(f"Rows lost: \t{len(bolig_data) - len(bolig_data_multiple_sales)}")
    return (bolig_data_multiple_sales,)


@app.cell
def _():
    return


@app.cell(column=5, hide_code=True)
def _(mo):
    mo.md(r"""
    # Turbines active or planned at house sale
    """)
    return


@app.cell
def _(bolig_data_multiple_sales):
    bolig_data_multiple_sales
    return


@app.cell
def _(pd, turbine_vejle_geo):
    # How many month ahead can we assume that the turbine have been publicly known and therefore affecting the sale
    _months_known = 24

    turbine_vejle_geo["known_date"] = turbine_vejle_geo[
        "Dato for oprindelig nettilslutning"
    ] - pd.DateOffset(months=12)
    turbine_vejle_geo
    return


@app.cell
def _(gpd, np, pd):
    def distance_to_planned_or_active_turbines_at_sale(
        house_sale: gpd.GeoSeries, turbines: gpd.GeoDataFrame
    ):
        sale_year = house_sale["Aar"]
        # 1. Create a boolean mask for the date range
        is_active = (turbines["known_date"] <= sale_year) & (
            turbines["Dato for afmeldning"] >= sale_year
        )

        # 2. Filter the GeoDataFrame
        active_turbines = turbines.loc[is_active]

        if active_turbines.empty:
            return pd.Series({"dist": np.nan, "turbine_id": np.nan})

        # 3. Calculate distances
        distances = active_turbines.distance(house_sale.geometry)

        # 4. Find the minimum distance and the index of that turbine
        min_dist = distances.min()
        closest_idx = distances.idxmin()

        # Get the actual ID value from the turbine table using the index
        turbine_id = active_turbines.loc[closest_idx, "id"]

        return pd.Series({"dist": min_dist, "turbine_id": turbine_id})
    return (distance_to_planned_or_active_turbines_at_sale,)


@app.cell
def _(
    bolig_data_multiple_sales,
    distance_to_planned_or_active_turbines_at_sale,
    turbine_vejle_geo,
):
    _results = bolig_data_multiple_sales.apply(
        lambda row: distance_to_planned_or_active_turbines_at_sale(
            row, turbine_vejle_geo
        ),
        axis=1,
    )

    # Join the results back to your main dataframe
    bolig_data_multiple_sales[["turbine_closest_dist", "turbine_closest_id"]] = (
        _results
    )
    return


@app.cell
def _(bolig_data_multiple_sales):
    bolig_data_multiple_sales
    return


@app.cell
def _(bolig_data_multiple_sales, pd):
    pd.options.display.max_rows = None
    print(bolig_data_multiple_sales.dtypes)

    return


@app.cell
def _(bolig_data_multiple_sales):
    bolig_ml = bolig_data_multiple_sales[
        [
            "VURderetAreal",
            "EjendomvaerdiBeloeb",
            "GrundvaerdiBeloeb",
            "turbine_closest_dist",
            "SamletKoebesum",
        ]
    ].copy()

    bolig_ml.dropna(
        subset=["turbine_closest_dist", "SamletKoebesum"], inplace=True
    )
    bolig_ml.isna().sum()
    len(bolig_ml)
    return (bolig_ml,)


@app.cell
def _(bolig_ml):
    bolig_ml
    return


@app.cell
def _():
    return


@app.cell(column=6, hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression
    """)
    return


@app.cell
def _():
    from sklearn.linear_model import (
        LinearRegression,
        SGDRegressor,
        RidgeCV,
        LassoCV,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        RandomizedSearchCV,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
    return (
        DecisionTreeRegressor,
        KNeighborsRegressor,
        LinearRegression,
        RandomForestRegressor,
        RandomizedSearchCV,
        SGDRegressor,
        cross_val_score,
        mean_squared_error,
        r2_score,
        train_test_split,
    )


@app.cell
def _(bolig_ml, train_test_split):
    X = bolig_ml.drop(columns=["SamletKoebesum"])
    y = bolig_ml["SamletKoebesum"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(
    LinearRegression,
    X,
    X_test,
    X_train,
    cross_val_score,
    np,
    y,
    y_test,
    y_train,
):
    _lin_reg = LinearRegression()
    _lin_reg.fit(X_train, y_train)
    _y_pred = _lin_reg.predict(X_test)
    _cv_score = cross_val_score(
        _lin_reg, X, y, cv=10, scoring="neg_mean_squared_error"
    )
    _r_squared = _lin_reg.score(X_test, y_test)
    # _mse = mean_squared_error(_y_pred, y_test)
    _avg_rmse = np.sqrt(-_cv_score.mean())

    print(f"R-squared: {_r_squared}")
    print(f"Avg. RMSE: {np.sqrt(_avg_rmse)}")
    print(f"Coef: {_lin_reg.coef_}")
    return


@app.cell
def _(
    DecisionTreeRegressor,
    KNeighborsRegressor,
    LinearRegression,
    RandomForestRegressor,
    SGDRegressor,
    X,
    X_test,
    X_train,
    cross_val_score,
    np,
    r2_score,
    y,
    y_test,
    y_train,
):
    models = {
        "linear regression": LinearRegression(),
        "SGD regression": SGDRegressor(),
        "KNN regression": KNeighborsRegressor(),
        "Decision tree regression": DecisionTreeRegressor(),
        "RF regression": RandomForestRegressor(),
    }
    for name, model in models.items():
        print(f"Processing {name}")
        model.fit(X_train, y_train)
        _y_pred = model.predict(X_test)
        _cv_score = cross_val_score(
            model, X, y, cv=10, scoring="neg_mean_squared_error"
        )
        _avg_rmse = np.sqrt(-_cv_score.mean())

        # 1. Get the standard R-squared
        _r2 = r2_score(y_test, _y_pred)

        # # 2. Get n and p
        # _n = X_test.shape[0]  # Number of samples
        # _p = X_test.shape[1]  # Number of features

        # # 3. Calculate Adjusted R-squared
        # _adj_r2 = 1 - (1 - _r2) * (_n - 1) / (_n - _p - 1)

        print(f"R-squared: {_r2}")
        # print(f"Adjusted R-squared: {_adj_r2}")
        print(f"Avg. RMSE: {_avg_rmse}")
        print("\n\n")
    return


@app.cell
def _(
    RandomForestRegressor,
    RandomizedSearchCV,
    X,
    X_test,
    X_train,
    mean_squared_error,
    np,
    pd,
    plt,
    sns,
    y_test,
    y_train,
):
    # 1. Update parameter distribution for Random Forest
    # We add 'n_estimators' (number of trees)
    _param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [
            1.0,
            "sqrt",
            "log2",
        ],  # 1.0 is the same as None (all features)
        "bootstrap": [True, False],
    }

    # 2. Create RandomForestRegressor
    _rf_reg = RandomForestRegressor(random_state=42)

    # 3. Create RandomizedSearchCV
    _random_search = RandomizedSearchCV(
        _rf_reg,
        param_distributions=_param_dist,  # Changed variable name to match sklearn param
        n_iter=20,  # Adjusted iterations for speed
        cv=5,  # 5-fold is usually enough and faster than 10
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    # 4. Fit the model
    _random_search.fit(X_train, y_train)

    # 5. Evaluation
    print(f"Best parameters: {_random_search.best_params_}")

    # Calculate metrics
    _y_pred = _random_search.predict(X_test)
    _test_r2 = _random_search.score(X_test, y_test)
    _test_rmse = np.sqrt(mean_squared_error(y_test, _y_pred))

    print(f"\nTest R-squared: {_test_r2}")
    print(f"Test RMSE: {_test_rmse}")



    # 1. Get feature importances from the best model
    _importances = _random_search.best_estimator_.feature_importances_
    _feature_names = X.columns # Assumes X is a pandas DataFrame

    # 2. Create a DataFrame for easy plotting
    _feature_importance_df = pd.DataFrame({
        'Feature': _feature_names,
        'Importance': _importances
    }).sort_values(by='Importance', ascending=False)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=_feature_importance_df, palette='viridis')
    plt.title('Feature Importance - Random Forest Regressor')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(column=7)
def _():
    return


if __name__ == "__main__":
    app.run()
