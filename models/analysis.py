import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely import wkt
    import altair as alt
    import contextily as cx
    from typing import Literal, Optional
    from models.data_handler import DataHandler, Turbines, Houses
    from models.data_handler import Preprocessor



@app.cell
def _():
    turbines = Turbines()
    return (turbines,)


@app.cell
def _():
    data_handler = DataHandler()

    data_handler_data: gpd.GeoDataFrame = data_handler.get_data("next")
    # data_handler_data
    return (data_handler_data,)


@app.cell
def _(turbines):
    turbines.gdf
    return


@app.cell
def _(data_handler_data: gpd.GeoDataFrame, turbines):
    data_with_turbines = data_handler_data[
        data_handler_data["has_new_turbine"] == 1
    ]
    turbines.gdf["is_active"] = turbines.gdf["afmelding_dato"].isna()

    data_with_turbines.set_geometry(data_handler_data.centroid, inplace=True)
    return (data_with_turbines,)


@app.cell
def _(data_with_turbines, turbines):
    fig, ax = plt.subplots(figsize=(10, 10))

    turbines.gdf.query("is_active").plot(
        ax=ax,
        legend=True,
        color="red",
        marker="^",
        markersize=60,
        label="Active Turbines",
    )
    turbines.gdf.query("not is_active").plot(
        ax=ax,
        legend=True,
        color="black",
        marker="^",
        markersize=60,
        alpha=0.6,
        label="Deactivated Turbines",
    )

    # Plot the spatial data, coloring points by the log_price target
    # Temporarily set geometry to centroids to plot the exact points
    data_with_turbines.plot(
        ax=ax,
        column="growth_rate",
        cmap="viridis",
        legend=True,
        markersize=80,  # Reduced marker size for point visibility
        alpha=0.6,
    )

    # Plot Vejle Station for reference using its coordinates
    # ax.scatter(
    #     x=533707.680,
    #     y=6173626.640,
    #     color="blue",
    #     marker="x",
    #     s=200,
    #     label="Vejle Station",
    # )

    # Add the basemap
    cx.add_basemap(
        ax, crs=data_with_turbines.crs, source=cx.providers.CartoDB.Positron
    )

    # Get the bounding box of your house sales data
    minx, miny, maxx, maxy = data_with_turbines.total_bounds

    # Define a buffer distance in meters (e.g., 2000 meters / 2km)
    buffer = 3000

    # Apply the buffer to the plot limits
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("House Sales Data Points")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
