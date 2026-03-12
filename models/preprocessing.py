import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")

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


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    # Data Display
    """)
    return


@app.cell
def _():
    from data_handler import DataHandler
    from data_handler import Turbines

    data_handler = DataHandler()
    turbines = Turbines()

    data_handler_data: gpd.GeoDataFrame = data_handler.get_data("next")
    # data_handler_data
    return data_handler_data, turbines


@app.cell
def _(data_handler_data: gpd.GeoDataFrame):
    data_handler_data[data_handler_data["has_new_turbine"] == 1]
    return


@app.cell
def _(turbines):
    turbines.gdf
    return


@app.cell
def _(data_handler_data: gpd.GeoDataFrame, turbines):
    sold_houses_with_turbines = data_handler_data[
        data_handler_data["has_new_turbine"] == 1
    ]
    turbines.gdf["is_active"] = turbines.gdf["afmelding_dato"].isna()

    turbines.gdf.set_geometry(
        turbines.gdf.centroid, inplace=True
    )

    sold_houses_with_turbines.set_geometry(
        data_handler_data.centroid, inplace=True
    )
    return (sold_houses_with_turbines,)


@app.cell
def _(sold_houses_with_turbines):
    sold_houses_with_turbines
    return


@app.cell
def _(sold_houses_with_turbines, turbines):
    _fig, _ax = plt.subplots(figsize=(20, 25))

    turbines.gdf.query("is_active").plot(
        ax=_ax,
        legend=True,
        color="red",
        marker="^",
        markersize=60,
        label="Aktiveret Vindmølle",
    )
    turbines.gdf.query("not is_active").plot(
        ax=_ax,
        legend=True,
        color="black",
        marker="^",
        markersize=60,
        alpha=0.6,
        label="Deaktiveret Vindmølle",
    )


    # Plot the spatial data, coloring points by the log_price target
    # Temporarily set geometry to centroids to plot the exact points
    sold_houses_with_turbines.plot(
        ax=_ax,
        column="growth_rate",
        cmap="viridis",
        legend_kwds={"label": "Pris Forskel %"},
        legend=True,
        # marker="x",
        markersize=150,  # Reduced marker size for point visibility
        alpha=0.6,
        label="Hus",
    )


    # Add the basemap
    cx.add_basemap(
        _ax,
        crs=sold_houses_with_turbines.crs,
        source=cx.providers.CartoDB.Positron,
    )

    # Get the bounding box of your house sales data
    _minx, _miny, _maxx, _maxy = sold_houses_with_turbines.total_bounds

    # Define a buffer distance in meters (e.g., 2000 meters / 2km)
    _buffer = 4000

    # Apply the buffer to the plot limits
    _ax.set_xlim(_minx - _buffer, _maxx + _buffer)
    _ax.set_ylim(_miny - _buffer, _maxy + _buffer)

    _ax.set_xticks([])
    _ax.set_yticks([])
    plt.title("Husprisers påvirkning af vindmøller i Vejle Kommune")
    plt.legend()


    # Select a subset of your data to display (e.g., top 5 rows and specific columns)
    _columns_to_show = [
        "adresse",
        "dist_to_new_turbine",
        "sale_year_prev",
        "SamletKoebesum_prev",
        "sale_year",
        "SamletKoebesum",
        "growth_rate",
        "price_change",
    ]  # Replace with your actual column names

    _table_data = sold_houses_with_turbines[_columns_to_show].head(20).copy()
    # _table_data = sold_houses_with_turbines[_columns_to_show]
    _table_data["growth_rate"] = _table_data["growth_rate"].round(2)
    _table_data["dist_to_new_turbine"] = _table_data["dist_to_new_turbine"].astype(
        int
    )
    _table_data["SamletKoebesum_prev"] = _table_data["SamletKoebesum_prev"].astype(
        int
    )
    _table_data["SamletKoebesum"] = _table_data["SamletKoebesum"].astype(int)

    _table_data.rename(
        columns={
            "adresse": "Adresse",
            "SamletKoebesum": "Pris 1",
            "SamletKoebesum_prev": "Pris 2",
            "growth_rate": "Prisudvikling (%)",
            "price_change": "Prisudvikling (DKK)",
            "dist_to_new_turbine": "Afstand til vindmølle (m)",
            "sale_year_prev": "Salgsår 1",
            "sale_year": "Salgsår 2",
        },
        inplace=True,
    )

    # Create the table
    _table = plt.table(
        cellText=_table_data.astype(
            str
        ).values,  # Convert to string to handle geometry objects safely
        colLabels=_table_data.columns,
        loc="bottom",
        cellLoc="right",
        bbox=[0, -0.5, 1.3, 0.48],
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    _table.auto_set_column_width(col=list(range(len(_table_data.columns))))
    # Add this to force the text color to black
    for _key, _cell in _table.get_celld().items():
        _cell.get_text().set_color("black")
        _cell.PAD = (
            0.05  # Gives the text a little breathing room from the cell walls
        )

    # Make room for the table below the plot
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(sold_houses_with_turbines, turbines):
    _fig, _ax = plt.subplots(figsize=(20, 25))

    turbines.gdf.query("is_active").plot(
        ax=_ax,
        legend=True,
        color="red",
        marker="^",
        markersize=60,
        label="Aktiveret Vindmølle",
    )
    turbines.gdf.query("not is_active").plot(
        ax=_ax,
        legend=True,
        color="black",
        marker="^",
        markersize=60,
        alpha=0.6,
        label="Deaktiveret Vindmølle",
    )

    _houses_5km_area = sold_houses_with_turbines.geometry.buffer(5000).union_all()
    _turbines_near_houses = turbines.gdf[
        turbines.gdf.geometry.intersects(_houses_5km_area)
    ].query("is_active")
    _turbine_radius = _turbines_near_houses.geometry.buffer(5000)

    _turbine_radius.plot(
        ax=_ax,
        facecolor="none",
        edgecolor="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.3,
    )

    _ax.plot(
        [],
        [],
        color="red",
        linewidth=1.5,
        linestyle="--",
        alpha=0.3,
        label="5km Radius fra aktiverede vindmøller",
    )

    # Plot the spatial data, coloring points by the log_price target
    # Temporarily set geometry to centroids to plot the exact points
    sold_houses_with_turbines.plot(
        ax=_ax,
        column="growth_rate",
        cmap="viridis",
        legend_kwds={"label": "Prisudvikling %"},
        legend=True,
        # marker="x",
        markersize=150,  # Reduced marker size for point visibility
        alpha=0.6,
        label="Hus",
    )


    # Add the basemap
    cx.add_basemap(
        _ax,
        crs=sold_houses_with_turbines.crs,
        source=cx.providers.CartoDB.Positron,
    )

    # Get the bounding box of your house sales data
    _minx, _miny, _maxx, _maxy = sold_houses_with_turbines.total_bounds

    # Define a buffer distance in meters (e.g., 2000 meters / 2km)
    _buffer = 5000

    # Apply the buffer to the plot limits
    _ax.set_xlim(_minx - _buffer, _maxx + _buffer)
    _ax.set_ylim(_miny - _buffer, _maxy + _buffer)

    _ax.set_xticks([])
    _ax.set_yticks([])
    plt.title("Husprisers påvirkning af vindmøller i Vejle Kommune")
    plt.legend()


    # Select a subset of your data to display (e.g., top 5 rows and specific columns)
    _columns_to_show = [
        "adresse",
        "dist_to_new_turbine",
        "sale_year_prev",
        "SamletKoebesum_prev",
        "sale_year",
        "SamletKoebesum",
        "growth_rate",
        "price_change",
    ]  # Replace with your actual column names

    _table_data = sold_houses_with_turbines[_columns_to_show].head(20).copy()
    # _table_data = sold_houses_with_turbines[_columns_to_show]
    _table_data["growth_rate"] = _table_data["growth_rate"].round(2)
    _table_data["dist_to_new_turbine"] = _table_data["dist_to_new_turbine"].astype(
        int
    )
    _table_data["SamletKoebesum_prev"] = _table_data["SamletKoebesum_prev"].astype(
        int
    )
    _table_data["SamletKoebesum"] = _table_data["SamletKoebesum"].astype(int)

    _table_data.rename(
        columns={
            "adresse": "Adresse",
            "SamletKoebesum": "Pris 1",
            "SamletKoebesum_prev": "Pris 2",
            "growth_rate": "Prisudvikling (%)",
            "price_change": "Prisudvikling (DKK)",
            "dist_to_new_turbine": "Afstand til vindmølle (m)",
            "sale_year_prev": "Salgsår 1",
            "sale_year": "Salgsår 2",
        },
        inplace=True,
    )

    # Create the table
    _table = plt.table(
        cellText=_table_data.astype(
            str
        ).values,  # Convert to string to handle geometry objects safely
        colLabels=_table_data.columns,
        loc="bottom",
        cellLoc="right",
        bbox=[0, -0.5, 1.3, 0.48],
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    _table.auto_set_column_width(col=list(range(len(_table_data.columns))))
    # Add this to force the text color to black
    for _key, _cell in _table.get_celld().items():
        _cell.get_text().set_color("black")
        _cell.PAD = (
            0.05  # Gives the text a little breathing room from the cell walls
        )

    # Make room for the table below the plot
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(sold_houses_with_turbines, turbines):
    _fig, _ax = plt.subplots(figsize=(10, 10))

    turbines.gdf.geometry = turbines.gdf.geometry.buffer(500)
    turbines.gdf.query("is_active").plot(
        ax=_ax,
        legend=True,
        color="red",
        # marker="^",
        # markersize=60,
        alpha=0.1,
        label="Active Turbines",
    )
    # turbines.gdf.query("not is_active").plot(
    #     ax=ax,
    #     legend=True,
    #     color="black",
    #     marker="^",
    #     markersize=60,
    #     alpha=0.6,
    #     label="Deactivated Turbines",
    # )


    # Plot the spatial data, coloring points by the log_price target
    # Temporarily set geometry to centroids to plot the exact points
    sold_houses_with_turbines.plot(
        ax=_ax,
        column="growth_rate",
        cmap="viridis",
        legend=True,
        markersize=50,  # Reduced marker size for point visibility
        alpha=1,
        label="House",
    )


    # Add the basemap
    cx.add_basemap(
        _ax,
        crs=sold_houses_with_turbines.crs,
        source=cx.providers.CartoDB.Positron,
    )

    # Get the bounding box of your house sales data
    minx, miny, maxx, maxy = sold_houses_with_turbines.total_bounds

    # Define a buffer distance in meters (e.g., 2000 meters / 2km)
    buffer = 5000

    # Apply the buffer to the plot limits
    _ax.set_xlim(minx - buffer, maxx + buffer)
    _ax.set_ylim(miny - buffer, maxy + buffer)

    _ax.set_xticks([])
    _ax.set_yticks([])
    plt.title("House Sales Data Points")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    # sales_next = get_comparative_sales_with_turbine(
    #     on="next", months_of_effect=24, radius_m=5000
    # )
    # sales_next
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    # sales_all = get_comparative_sales_with_turbine(
    #     on="all", months_of_effect=24, radius_m=5000
    # )
    # sales_all.drop(
    #     columns=[
    #         "geometry",
    #         "salgs_dato",
    #         "salgs_dato_prev",
    #         "vurderingsaar",
    #         "byg038SamletBygningsAreal_prev",
    #         "byg039BygningensSamlBoligAreal_prev",
    #         "house_geometry_original",
    #         "tilslutning_dato",
    #         "date_of_effect",
    #         "BFEnummer",
    #         "byg038SamletBygningsAreal",
    #         "byg039BygningensSamlBoligAreal",
    #         "GrundvaerdiBeloeb",
    #         "GrundvaerdiBeloeb_prev",
    #         "EjendomvaerdiBeloeb",
    #         "EjendomvaerdiBeloeb_prev",
    #         "SamletKoebesum_prev",
    #         "VURderetAreal",
    #         "VURderetAreal_prev",
    #     ],
    #     inplace=True,
    # )
    # sales_all.dropna(inplace=True)
    # sales_all
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    # Primary Function
    """)
    return


@app.cell
def _():
    # def get_comparative_sales_with_turbine(
    #     on: Literal["next", "all"],
    #     months_of_effect: int = 24,
    #     radius_m: int = 5000,
    # ) -> gpd.GeoDataFrame:
    #     house_sales = HouseSales()
    #     turbines = Turbines()
    #     comparative_sales = ComparativeHouseSales(house_sales=house_sales.gdf)
    #     data_processor = Preprocessor(
    #         house_sales=comparative_sales.compare(on=on), turbines=turbines.gdf
    #     )

    #     return data_processor.join_nearest_activated_turbine(
    #         months_of_effect=months_of_effect, buffer_radius_m=radius_m
    #     )
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    # House Sales Classes
    """)
    return


@app.cell(hide_code=True)
def _():
    # class HouseSales:
    #     def __init__(self, file_path: str = "data/boligsalg.csv"):
    #         self._df = self._load_data(path=file_path)
    #         self.gdf = self._to_geodataframe()
    #         self._rename_cols()
    #         self._handle_datetime()
    #         self._drop_cols()

    #         self.gdf_multiple_sales = self.get_houses_with_multiple_sales()

    #     def _load_data(self, path: str, csv_sep=";") -> pd.DataFrame:
    #         return pd.read_csv(path, sep=csv_sep)

    #     def _to_geodataframe(self) -> gpd.GeoDataFrame:
    #         _geometry_column = "byg404Koordinat"
    #         self._df[_geometry_column] = self._df[_geometry_column].apply(
    #             wkt.loads
    #         )

    #         _data_gdf = gpd.GeoDataFrame(
    #             self._df,
    #             geometry=_geometry_column,
    #             crs="EPSG:25832",
    #         )

    #         return _data_gdf

    #     def _rename_cols(self) -> None:
    #         self.gdf.rename_geometry("geometry", inplace=True)
    #         self.gdf.rename(
    #             columns={"KoebsaftaleDato": "salgs_dato", "Aar": "vurderingsaar"},
    #             inplace=True,
    #         )

    #     def _handle_datetime(self) -> None:
    #         self.gdf["vurderingsaar"] = pd.to_datetime(
    #             self.gdf["vurderingsaar"], format="%Y"
    #         )
    #         self.gdf["salgs_dato"] = pd.to_datetime(self.gdf["salgs_dato"])

    #     def _drop_cols(self, columns: list[str] = None):
    #         self.__drop_eur_currency_code()
    #         if columns:
    #             self.gdf.drop(columns=columns, inplace=True)

    #     def __drop_eur_currency_code(self):
    #         return self.gdf.drop(
    #             self.gdf[self.gdf["Valutakode"] == "EUR"].index, inplace=True
    #         )

    #     def get_houses_with_multiple_sales(self) -> gpd.GeoDataFrame:
    #         """
    #         Get houses that have been sold multiple times, and drop rows without sale price
    #         """
    #         _bfe_count = self.gdf["BFEnummer"].value_counts()
    #         _bfe_more_than_two = _bfe_count[_bfe_count > 1].index
    #         gdf_multiple_sales = self.gdf[
    #             self.gdf["BFEnummer"].isin(_bfe_more_than_two)
    #         ].copy()
    #         gdf_multiple_sales.dropna(subset=["SamletKoebesum"], inplace=True)

    #         return gdf_multiple_sales
    return


@app.cell(hide_code=True)
def _():
    # class ComparativeHouseSales:
    #     def __init__(self, house_sales: gpd.GeoDataFrame):
    #         self.house_sales = house_sales

    #     def compare(
    #         self, on: Optional[Literal["next", "all"]]
    #     ) -> gpd.GeoDataFrame:
    #         """
    #         Compare sales of the same house to analyze price development.
    #         Parameters:
    #         - on: "next" to compare each sale with the immediately previous sale, or
    #               "all" to compare each sale with all previous sales (powerset).
    #         Returns:
    #         - A DataFrame with growth rates and time differences between sales.
    #         """
    #         if on not in ["next", "all"]:
    #             raise ValueError('Only "next" and "all" are valid parameters')

    #         if on == "next":
    #             return self.__join_on_next()
    #         elif on == "all":
    #             return self.__join_on_all()

    #     def __join_on_all(self) -> gpd.GeoDataFrame:
    #         """
    #         Make a Powerset of all sales on the same house
    #         This could be used to compare prices over longer periods, to ensure more data where sale 1 doesn't have a turbine, and sale 2 does.
    #         """
    #         # Perform a self-merge on BFEnummer
    #         _df_all_pairs = pd.merge(
    #             self.house_sales,
    #             self.house_sales,
    #             on="BFEnummer",
    #             suffixes=("", "_prev"),
    #         )

    #         # Filter to keep only rows where the current sale date is after the previous sale date
    #         _df_all_pairs = _df_all_pairs[
    #             _df_all_pairs["salgs_dato"] > _df_all_pairs["salgs_dato_prev"]
    #         ]

    #         # Calculate growth rate and time difference
    #         _df_all_pairs["growth_rate"] = (
    #             _df_all_pairs["SamletKoebesum"]
    #             - _df_all_pairs["SamletKoebesum_prev"]
    #         ) / _df_all_pairs["SamletKoebesum_prev"]

    #         _df_all_pairs["years_diff"] = (
    #             _df_all_pairs["salgs_dato"] - _df_all_pairs["salgs_dato_prev"]
    #         ).dt.days / 365.25

    #         return _df_all_pairs.copy()

    #     def __join_on_next(self) -> gpd.GeoDataFrame:
    #         """
    #                 Compare a sale with the previous sale
    #         This will give a more accurate depiction of price development.
    #         """

    #         # 1. Sort and assign a "Rank" to each sale (0 for first sale, 1 for second, etc.)
    #         _df = self.house_sales.sort_values(
    #             by=["BFEnummer", "salgs_dato"]
    #         ).copy()
    #         _df["sale_rank"] = _df.groupby("BFEnummer").cumcount()

    #         # 2. Perform the merge strictly on (BFEnummer) and (Rank vs Rank-1)
    #         # We align the "current" sale_rank with the "previous" sale_rank (which is current - 1)
    #         _house_sale_next_compare = pd.merge(
    #             _df,
    #             _df,
    #             left_on=["BFEnummer", "sale_rank"],
    #             right_on=[
    #                 "BFEnummer",
    #                 _df["sale_rank"] + 1,
    #             ],  # Join Sale N with Sale N-1
    #             suffixes=("", "_prev"),
    #         )

    #         # 3. Calculate metrics (same as before)
    #         _house_sale_next_compare["growth_rate"] = (
    #             _house_sale_next_compare["SamletKoebesum"]
    #             - _house_sale_next_compare["SamletKoebesum_prev"]
    #         ) / _house_sale_next_compare["SamletKoebesum_prev"]

    #         _house_sale_next_compare["years_diff"] = (
    #             _house_sale_next_compare["salgs_dato"]
    #             - _house_sale_next_compare["salgs_dato_prev"]
    #         ).dt.days / 365.25

    #         # 4. Cleanup (drop the helper rank columns)
    #         house_sale_next_compare = _house_sale_next_compare.drop(
    #             columns=["sale_rank", "sale_rank_prev"]
    #         )

    #         return house_sale_next_compare
    return


@app.cell(column=5, hide_code=True)
def _():
    mo.md(r"""
    # Turbine Class
    """)
    return


@app.cell(hide_code=True)
def _():
    # class Turbines:
    #     def __init__(
    #         self,
    #         file_path: str = "data/Vindmølledata til 2025-01.xlsx",
    #         kommune_code: int = 630,
    #     ):
    #         self._df = self._load_data(path=file_path)
    #         self._df = self._filter_by_kommune(kommune_code)
    #         self.gdf = self._to_geodataframe()
    #         self._rename_cols()
    #         self._handle_datetime()
    #         self._drop_cols()

    #     def _load_data(self, path: str, skip_rows: int = 10) -> pd.DataFrame:
    #         return pd.read_excel(path, skiprows=skip_rows)

    #     def _to_geodataframe(self) -> gpd.GeoDataFrame:
    #         self._df.columns = self._df.columns.astype(str)

    #         _gdf = gpd.GeoDataFrame(
    #             self._df[
    #                 [
    #                     "Møllenummer (GSRN)",
    #                     "X (øst) koordinat \nUTM 32 Euref89",
    #                     "Y (nord) koordinat \nUTM 32 Euref89",
    #                     "Dato for oprindelig nettilslutning",
    #                     "Dato for afmeldning",
    #                     "Koordinatoprindelse",
    #                     "Rotor-diameter (m)",
    #                     "Navhøjde (m)",
    #                 ]
    #             ],
    #             geometry=gpd.points_from_xy(
    #                 x=self._df["X (øst) koordinat \nUTM 32 Euref89"],
    #                 y=self._df["Y (nord) koordinat \nUTM 32 Euref89"],
    #                 crs="EPSG:25832",
    #             ),
    #         )

    #         return _gdf

    #     def _rename_cols(self):
    #         self.gdf.rename(
    #             columns={
    #                 "X (øst) koordinat \nUTM 32 Euref89": "x",
    #                 "Y (nord) koordinat \nUTM 32 Euref89": "y",
    #                 "Møllenummer (GSRN)": "id",
    #                 "Rotor-diameter (m)": "rotor_diameter_m",
    #                 "Navhøjde (m)": "height_pole_m",
    #                 "Dato for afmeldning": "afmelding_dato",
    #                 "Dato for oprindelig nettilslutning": "tilslutning_dato",
    #             },
    #             inplace=True,
    #         )

    #     def _handle_datetime(self) -> None:
    #         self.gdf["afmelding_dato"] = pd.to_datetime(self.gdf["afmelding_dato"])
    #         self.gdf["tilslutning_dato"] = pd.to_datetime(
    #             self.gdf["tilslutning_dato"]
    #         )

    #     def _drop_cols(self, columns: list[str] = None):
    #         self.gdf.dropna(subset=["x", "y"], inplace=True)

    #     def _filter_by_kommune(self, kommune_kode: int = 630):
    #         return self._df[self._df["Kommune"].str.contains(str(kommune_kode))]
    return


@app.cell(column=6, hide_code=True)
def _():
    mo.md(r"""
    # Data Processor
    """)
    return


@app.cell
def _():
    # class Preprocessor:
    #     def __init__(
    #         self, house_sales: gpd.GeoDataFrame, turbines: gpd.GeoDataFrame
    #     ):
    #         self.house_sales = house_sales
    #         self.turbines = turbines

    #     def join_nearest_activated_turbine(
    #         self,
    #         buffer_radius_m: int = 5000,
    #         months_of_effect: int = 24,
    #     ) -> gpd.GeoDataFrame:
    #         """For each house sale, find the nearest turbine that was activated in the relevant time window. This involves several steps:
    #         1. Buffer the house points to create a search area.
    #         2. Perform a spatial join to find turbines within the buffer.
    #         3. Filter turbines based on the activation date relative to the sale date.
    #         4. Calculate the distance to the nearest valid turbine and merge this information back to the original house sales dataframe.


    #         Parameters:
    #         - buffer_radius_m: The radius in meters to search for turbines around each house sale.
    #         - months_of_effect: The number of months to offset the turbine activation date to account for pre-activation effects on house prices.

    #         Returns:
    #         - A GeoDataFrame with the nearest activated turbine information merged for each house sale.
    #         """

    #         # 1. Prepare Turbines (Offset dates)
    #         turbines = self._offset_impact(
    #             turbines=self.turbines.copy(), months=months_of_effect
    #         )

    #         houses_search = self.house_sales.copy()

    #         # 2. Find Candidates (Spatial Join)
    #         candidates = self._get_turbines_within_radius(
    #             turbines=turbines,
    #             house_sales=houses_search,
    #             radius_m=buffer_radius_m,
    #         )

    #         # 3. Filter Candidates (Temporal Filter)
    #         valid_candidates = self._filter_active_turbines(turbines=candidates)

    #         # 4. Get Nearest (Deduplicate)
    #         nearest_new_turbine = self._get_nearest_turbine(
    #             candidates=valid_candidates
    #         )

    #         # 5. Merge Result: LEFT JOIN [CRITICAL CHANGE]
    #         # We use 'left' to keep houses even if they found NO valid turbine
    #         final_df = houses_search.join(
    #             nearest_new_turbine[
    #                 [
    #                     "dist_to_new_turbine",
    #                     "tilslutning_dato",
    #                     "date_of_effect",
    #                     "rotor_diameter_m",
    #                     "height_pole_m",
    #                 ]
    #             ],
    #             rsuffix="_turb",
    #             how="left",
    #         )

    #         # 6. Handle Control Group (Fill NaNs) [CRITICAL CHANGE]

    #         # A. Create a binary flag: 1 = Treated (Turbine appeared), 0 = Control (No turbine)
    #         final_df["has_new_turbine"] = (
    #             final_df["dist_to_new_turbine"].notna().astype(int)
    #         )

    #         # B. Fill Distance: Set 'No Turbine' to a large number (e.g., 2x radius)
    #         # This tells the model: "This house is effectively infinitely far from a new turbine"
    #         # This preserves the slope: Small Distance = High Impact, Large Distance = Low Impact.
    #         fill_distance = buffer_radius_m * 2
    #         final_df["dist_to_new_turbine"] = final_df[
    #             "dist_to_new_turbine"
    #         ].fillna(fill_distance)

    #         # C. Fill Technical Specs: Set to 0 implies no visual impact
    #         final_df["rotor_diameter_m"] = final_df["rotor_diameter_m"].fillna(0)
    #         final_df["height_pole_m"] = final_df["height_pole_m"].fillna(0)

    #         # 7. Final Cleanup
    #         final_df = self._drop_rows_without_sale_price(gdf=final_df)
    #         final_df = self._drop_cols(gdf=final_df)

    #         # # Fill NaNs for houses where no new turbine appeared in the window
    #         # final_df = final_df.dropna(subset=["dist_to_new_turbine"])
    #         # final_df = self._drop_rows_without_sale_price(gdf=final_df)
    #         # final_df = self._drop_cols(gdf=final_df)

    #         return final_df

    #     def _offset_impact(self, turbines: gpd.GeoDataFrame, months: int = 24):
    #         """Offset the turbine activation date to account for the fact that turbine announcements and construction can affect house prices even before the turbine is fully operational. This creates a "date of effect" that is earlier than the actual activation date."""
    #         # Assume turbine activation would affect house prices even before activation date.
    #         # As soon as a turbine location has been announced, it is assumed to affect house prices
    #         turbines["date_of_effect"] = turbines[
    #             "tilslutning_dato"
    #         ] - pd.DateOffset(months=months)

    #         return turbines

    #     def _get_turbines_within_radius(
    #         self,
    #         turbines: gpd.GeoDataFrame,
    #         house_sales: gpd.GeoDataFrame,
    #         radius_m: int = 5000,
    #     ):
    #         """Find turbines within a certain radius of each house sale. This is done by buffering the house points and performing a spatial join with the turbine points. This significantly reduces the number of turbine-house pairs we need to consider in the temporal filtering step, improving performance."""
    #         # 2. Preserve Turbine Geometry: Save it to a column so it survives the join
    #         turbines["turb_geometry"] = turbines.geometry

    #         # 3. Create Search Area: Buffer houses (e.g., 5km) to limit the search
    #         # This significantly optimizes the join compared to a full cross-join
    #         house_sales["house_geometry_original"] = (
    #             house_sales.geometry
    #         )  # Keep original point
    #         house_sales["geometry"] = house_sales.geometry.buffer(radius_m)

    #         # 4. Spatial Join: Find all turbines within 5km of each house
    #         # We use indices to map back later
    #         candidates = gpd.sjoin(
    #             house_sales, turbines, how="inner", predicate="intersects"
    #         )

    #         return candidates

    #     def _get_nearest_turbine(
    #         self, candidates: gpd.GeoDataFrame
    #     ) -> gpd.GeoDataFrame:
    #         """
    #         For each house sale, find the nearest turbine that was activated in the relevant time window.
    #         """
    #         # 6. Calculate Distance: House Point <-> Turbine Point
    #         # We use the preserved geometries
    #         candidates["dist_to_new_turbine"] = candidates[
    #             "house_geometry_original"
    #         ].distance(candidates["turb_geometry"])

    #         # 7. Select Nearest: Sort by distance and keep only the closest per house sale row
    #         # Grouping by the index of the original houses dataframe ensures we map correctly
    #         nearest_new_turbine = (
    #             candidates.sort_values("dist_to_new_turbine")
    #             .groupby(level=0)
    #             .head(1)
    #         )

    #         return nearest_new_turbine

    #     def _filter_active_turbines(self, turbines: gpd.GeoDataFrame):
    #         """Filter turbines to ensure we only consider those that were active in the relevant time window for each sale."""
    #         # 5. Temporal Filter
    #         # Turbine must be activated BETWEEN the previous sale and current sale
    #         # And optionally, ensure it wasn't decommissioned before the current sale
    #         mask_turbine_activated_after_prev_sale = (
    #             turbines["date_of_effect"] > turbines["salgs_dato_prev"]
    #         )
    #         mask_turbine_activated_before_current_sale = (
    #             turbines["date_of_effect"] <= turbines["salgs_dato"]
    #         )
    #         mask_turbine_not_decommissioned_before_current_sale = turbines[
    #             "afmelding_dato"
    #         ].isna() | (turbines["afmelding_dato"] > turbines["salgs_dato"])

    #         return turbines[
    #             (
    #                 mask_turbine_activated_after_prev_sale
    #                 & mask_turbine_activated_before_current_sale
    #                 & mask_turbine_not_decommissioned_before_current_sale
    #             )
    #         ].copy()

    #     def _drop_cols(self, gdf: gpd.GeoDataFrame):
    #         """Drop columns that are not relevant for the ML model and could cause data leakage. This includes any columns related to the previous sale, as well as any columns that are not needed for the analysis."""
    #         return gdf.drop(
    #             columns=[
    #                 "Kommunekode",
    #                 "Kommunenavn",
    #                 "Postnr",
    #                 "Vejnavn",
    #                 "HusNr",
    #                 # "PostDistrikt",
    #                 "BenyttelseKode_T",
    #                 "BenyttelseKode",
    #                 "Valutakode",
    #                 "byg406Koordinatsystem_T",
    #                 "byg406Koordinatsystem",
    #                 "KontantKoebesum",
    #                 "Kommunekode_prev",
    #                 "BenyttelseKode_prev",
    #                 "Kommunenavn_prev",
    #                 "Postnr_prev",
    #                 "Vejnavn_prev",
    #                 "HusNr_prev",
    #                 "PostDistrikt_prev",
    #                 "BenyttelseKode_T_prev",
    #                 "BenyttelseKode_prev",
    #                 "Valutakode_prev",
    #                 "byg406Koordinatsystem_T_prev",
    #                 "byg406Koordinatsystem_prev",
    #                 "KontantKoebesum_prev",
    #                 "byg021BygningensAnvendelse",
    #                 "byg021BygningensAnvendelse_prev",
    #                 "geometry_prev",
    #                 "vurderingsaar_prev",
    #                 "byg026OpFoerelsesAar_prev",
    #             ],
    #             errors="ignore",
    #         )

    #     def _drop_rows_without_sale_price(self, gdf: gpd.GeoDataFrame):
    #         """Drop rows where the sale price is zero, as these do not represent actual sales and could skew the analysis."""
    #         _no_target = gdf[gdf["SamletKoebesum"] == 0]
    #         return gdf.drop(_no_target.index)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Prep Data for ML
    """)
    return


@app.cell(column=7)
def _():
    return


if __name__ == "__main__":
    app.run()
