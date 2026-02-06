import marimo

__generated_with = "0.19.8"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely import wkt
    import altair as alt

    from typing import Literal, Optional


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    # Data Display
    """)
    return


@app.cell
def _():
    sales_next = get_comparative_sales_with_turbine(
        on="next", months_of_effect=24, radius_m=5000
    )
    sales_next
    return


@app.cell
def _():
    sales_all = get_comparative_sales_with_turbine(
        on="all", months_of_effect=24, radius_m=5000
    )
    sales_all.drop(
        columns=[
            "geometry",
            "salgs_dato",
            "salgs_dato_prev",
            "vurderingsaar",
            "byg038SamletBygningsAreal_prev",
            "byg039BygningensSamlBoligAreal_prev",
            "house_geometry_original",
            "tilslutning_dato",
            "date_of_effect",
            "BFEnummer",
            "byg038SamletBygningsAreal",
            "byg039BygningensSamlBoligAreal",
            "GrundvaerdiBeloeb",
            "GrundvaerdiBeloeb_prev",
            "EjendomvaerdiBeloeb",
            "EjendomvaerdiBeloeb_prev",
            "SamletKoebesum_prev",
            "VURderetAreal",
            "VURderetAreal_prev",
        ],
        inplace=True,
    )
    sales_all.dropna(inplace=True)
    sales_all
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    # Primary Function
    """)
    return


@app.function
def get_comparative_sales_with_turbine(
    on: Literal["next", "all"],
    months_of_effect: int = 24,
    radius_m: int = 5000,
) -> gpd.GeoDataFrame:
    house_sales = HouseSales()
    turbines = Turbines()
    comparative_sales = ComparativeHouseSales(house_sales=house_sales.gdf)
    data_processor = Preprocessor(
        house_sales=comparative_sales.compare(on=on), turbines=turbines.gdf
    )

    return data_processor.join_nearest_activated_turbine(
        months_of_effect=months_of_effect, buffer_radius_m=radius_m
    )


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    # House Sales Classes
    """)
    return


@app.class_definition(hide_code=True)
class HouseSales:
    def __init__(self, file_path: str = "data/boligsalg.csv"):
        self._df = self._load_data(path=file_path)
        self.gdf = self._to_geodataframe()
        self._rename_cols()
        self._handle_datetime()
        self._drop_cols()

        self.gdf_multiple_sales = self.get_houses_with_multiple_sales()

    def _load_data(self, path: str, csv_sep=";") -> pd.DataFrame:
        return pd.read_csv(path, sep=csv_sep)

    def _to_geodataframe(self) -> gpd.GeoDataFrame:
        _geometry_column = "byg404Koordinat"
        self._df[_geometry_column] = self._df[_geometry_column].apply(wkt.loads)

        _data_gdf = gpd.GeoDataFrame(
            self._df,
            geometry=_geometry_column,
            crs="EPSG:25832",
        )

        return _data_gdf

    def _rename_cols(self) -> None:
        self.gdf.rename_geometry("geometry", inplace=True)
        self.gdf.rename(
            columns={"KoebsaftaleDato": "salgs_dato", "Aar": "vurderingsaar"},
            inplace=True,
        )

    def _handle_datetime(self) -> None:
        self.gdf["vurderingsaar"] = pd.to_datetime(
            self.gdf["vurderingsaar"], format="%Y"
        )
        self.gdf["salgs_dato"] = pd.to_datetime(self.gdf["salgs_dato"])

    def _drop_cols(self, columns: list[str] = None):
        self.__drop_eur_currency_code()
        if columns:
            self.gdf.drop(columns=columns, inplace=True)

    def __drop_eur_currency_code(self):
        return self.gdf.drop(
            self.gdf[self.gdf["Valutakode"] == "EUR"].index, inplace=True
        )

    def get_houses_with_multiple_sales(self) -> gpd.GeoDataFrame:
        """
        Get houses that have been sold multiple times, and drop rows without sale price
        """
        _bfe_count = self.gdf["BFEnummer"].value_counts()
        _bfe_more_than_two = _bfe_count[_bfe_count > 1].index
        gdf_multiple_sales = self.gdf[
            self.gdf["BFEnummer"].isin(_bfe_more_than_two)
        ].copy()
        gdf_multiple_sales.dropna(subset=["SamletKoebesum"], inplace=True)

        return gdf_multiple_sales


@app.class_definition(hide_code=True)
class ComparativeHouseSales:
    def __init__(self, house_sales: gpd.GeoDataFrame):
        self.house_sales = house_sales

    def compare(self, on: Optional[Literal["next", "all"]]) -> gpd.GeoDataFrame:
        """
        Compare sales of the same house to analyze price development.
        Parameters:
        - on: "next" to compare each sale with the immediately previous sale, or
              "all" to compare each sale with all previous sales (powerset).
        Returns:
        - A DataFrame with growth rates and time differences between sales.
        """
        if on not in ["next", "all"]:
            raise ValueError('Only "next" and "all" are valid parameters')

        if on == "next":
            return self.__join_on_next()
        elif on == "all":
            return self.__join_on_all()

    def __join_on_all(self) -> gpd.GeoDataFrame:
        """
        Make a Powerset of all sales on the same house
        This could be used to compare prices over longer periods, to ensure more data where sale 1 doesn't have a turbine, and sale 2 does.
        """
        # Perform a self-merge on BFEnummer
        _df_all_pairs = pd.merge(
            self.house_sales,
            self.house_sales,
            on="BFEnummer",
            suffixes=("", "_prev"),
        )

        # Filter to keep only rows where the current sale date is after the previous sale date
        _df_all_pairs = _df_all_pairs[
            _df_all_pairs["salgs_dato"] > _df_all_pairs["salgs_dato_prev"]
        ]

        # Calculate growth rate and time difference
        _df_all_pairs["growth_rate"] = (
            _df_all_pairs["SamletKoebesum"] - _df_all_pairs["SamletKoebesum_prev"]
        ) / _df_all_pairs["SamletKoebesum_prev"]

        _df_all_pairs["years_diff"] = (
            _df_all_pairs["salgs_dato"] - _df_all_pairs["salgs_dato_prev"]
        ).dt.days / 365.25

        return _df_all_pairs.copy()

    def __join_on_next(self) -> gpd.GeoDataFrame:
        """
                Compare a sale with the previous sale
        This will give a more accurate depiction of price development.
        """

        # 1. Sort and assign a "Rank" to each sale (0 for first sale, 1 for second, etc.)
        _df = self.house_sales.sort_values(by=["BFEnummer", "salgs_dato"]).copy()
        _df["sale_rank"] = _df.groupby("BFEnummer").cumcount()

        # 2. Perform the merge strictly on (BFEnummer) and (Rank vs Rank-1)
        # We align the "current" sale_rank with the "previous" sale_rank (which is current - 1)
        _house_sale_next_compare = pd.merge(
            _df,
            _df,
            left_on=["BFEnummer", "sale_rank"],
            right_on=[
                "BFEnummer",
                _df["sale_rank"] + 1,
            ],  # Join Sale N with Sale N-1
            suffixes=("", "_prev"),
        )

        # 3. Calculate metrics (same as before)
        _house_sale_next_compare["growth_rate"] = (
            _house_sale_next_compare["SamletKoebesum"]
            - _house_sale_next_compare["SamletKoebesum_prev"]
        ) / _house_sale_next_compare["SamletKoebesum_prev"]

        _house_sale_next_compare["years_diff"] = (
            _house_sale_next_compare["salgs_dato"]
            - _house_sale_next_compare["salgs_dato_prev"]
        ).dt.days / 365.25

        # 4. Cleanup (drop the helper rank columns)
        house_sale_next_compare = _house_sale_next_compare.drop(
            columns=["sale_rank", "sale_rank_prev"]
        )

        return house_sale_next_compare


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    # Turbine Class
    """)
    return


@app.class_definition(hide_code=True)
class Turbines:
    def __init__(
        self,
        file_path: str = "data/Vindmølledata til 2025-01.xlsx",
        kommune_code: int = 630,
    ):
        self._df = self._load_data(path=file_path)
        self._df = self._filter_by_kommune(kommune_code)
        self.gdf = self._to_geodataframe()
        self._rename_cols()
        self._handle_datetime()
        self._drop_cols()

    def _load_data(self, path: str, skip_rows: int = 10) -> pd.DataFrame:
        return pd.read_excel(path, skiprows=skip_rows)

    def _to_geodataframe(self) -> gpd.GeoDataFrame:
        self._df.columns = self._df.columns.astype(str)

        _gdf = gpd.GeoDataFrame(
            self._df[
                [
                    "Møllenummer (GSRN)",
                    "X (øst) koordinat \nUTM 32 Euref89",
                    "Y (nord) koordinat \nUTM 32 Euref89",
                    "Dato for oprindelig nettilslutning",
                    "Dato for afmeldning",
                    "Koordinatoprindelse",
                    "Rotor-diameter (m)",
                    "Navhøjde (m)",
                ]
            ],
            geometry=gpd.points_from_xy(
                x=self._df["X (øst) koordinat \nUTM 32 Euref89"],
                y=self._df["Y (nord) koordinat \nUTM 32 Euref89"],
                crs="EPSG:25832",
            ),
        )

        return _gdf

    def _rename_cols(self):
        self.gdf.rename(
            columns={
                "X (øst) koordinat \nUTM 32 Euref89": "x",
                "Y (nord) koordinat \nUTM 32 Euref89": "y",
                "Møllenummer (GSRN)": "id",
                "Rotor-diameter (m)": "rotor_diameter_m",
                "Navhøjde (m)": "height_pole_m",
                "Dato for afmeldning": "afmelding_dato",
                "Dato for oprindelig nettilslutning": "tilslutning_dato",
            },
            inplace=True,
        )

    def _handle_datetime(self) -> None:
        self.gdf["afmelding_dato"] = pd.to_datetime(self.gdf["afmelding_dato"])
        self.gdf["tilslutning_dato"] = pd.to_datetime(self.gdf["tilslutning_dato"])

    def _drop_cols(self, columns: list[str] = None):
        self.gdf.dropna(subset=["x", "y"], inplace=True)

    def _filter_by_kommune(self, kommune_kode: int = 630):
        return self._df[self._df["Kommune"].str.contains(str(kommune_kode))]


@app.cell(column=5, hide_code=True)
def _():
    mo.md(r"""
    # Data Processor
    """)
    return


@app.class_definition
class Preprocessor:
    def __init__(self, house_sales: gpd.GeoDataFrame, turbines: gpd.GeoDataFrame):
        self.house_sales = house_sales
        self.turbines = turbines

    def join_nearest_activated_turbine(
        self,
        buffer_radius_m: int = 5000,
        months_of_effect: int = 24,
    ) -> gpd.GeoDataFrame:
        """For each house sale, find the nearest turbine that was activated in the relevant time window. This involves several steps:
        1. Buffer the house points to create a search area.
        2. Perform a spatial join to find turbines within the buffer.
        3. Filter turbines based on the activation date relative to the sale date.
        4. Calculate the distance to the nearest valid turbine and merge this information back to the original house sales dataframe.


        Parameters:
        - buffer_radius_m: The radius in meters to search for turbines around each house sale.
        - months_of_effect: The number of months to offset the turbine activation date to account for pre-activation effects on house prices.

        Returns:
        - A GeoDataFrame with the nearest activated turbine information merged for each house sale.
        """

        turbines = self._offset_impact(
            turbines=self.turbines.copy(), months=months_of_effect
        )

        houses_search = self.house_sales.copy()

        candidates = self._get_turbines_within_radius(
            turbines=turbines, house_sales=houses_search
        )
        valid_candidates = self._filter_active_turbines(turbines=candidates)

        nearest_new_turbine = self._get_nearest_turbine(candidates=valid_candidates)

        # 8. Merge Result: Join the distance and turbine ID back to your original dataset
        final_df = houses_search.join(
            nearest_new_turbine[
                [
                    "dist_to_new_turbine",
                    "tilslutning_dato",
                    "date_of_effect",
                    "rotor_diameter_m",
                    "height_pole_m",
                ]
            ],
            rsuffix="_turb",
        )

        # Fill NaNs for houses where no new turbine appeared in the window
        final_df = final_df.dropna(subset=["dist_to_new_turbine"])
        final_df = self._drop_rows_without_sale_price(gdf=final_df)
        final_df = self._drop_cols(gdf=final_df)

        return final_df

    def _offset_impact(self, turbines: gpd.GeoDataFrame, months: int = 24):
        """Offset the turbine activation date to account for the fact that turbine announcements and construction can affect house prices even before the turbine is fully operational. This creates a "date of effect" that is earlier than the actual activation date."""
        # Assume turbine activation would affect house prices even before activation date.
        # As soon as a turbine location has been announced, it is assumed to affect house prices
        turbines["date_of_effect"] = turbines["tilslutning_dato"] - pd.DateOffset(
            months=months
        )

        return turbines

    def _get_turbines_within_radius(
        self,
        turbines: gpd.GeoDataFrame,
        house_sales: gpd.GeoDataFrame,
        radius_m: int = 5000,
    ):
        """Find turbines within a certain radius of each house sale. This is done by buffering the house points and performing a spatial join with the turbine points. This significantly reduces the number of turbine-house pairs we need to consider in the temporal filtering step, improving performance."""
        # 2. Preserve Turbine Geometry: Save it to a column so it survives the join
        turbines["turb_geometry"] = turbines.geometry

        # 3. Create Search Area: Buffer houses (e.g., 5km) to limit the search
        # This significantly optimizes the join compared to a full cross-join
        house_sales["house_geometry_original"] = (
            house_sales.geometry
        )  # Keep original point
        house_sales["geometry"] = house_sales.geometry.buffer(radius_m)

        # 4. Spatial Join: Find all turbines within 5km of each house
        # We use indices to map back later
        candidates = gpd.sjoin(
            house_sales, turbines, how="inner", predicate="intersects"
        )

        return candidates

    def _get_nearest_turbine(self, candidates: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        For each house sale, find the nearest turbine that was activated in the relevant time window.
        """
        # 6. Calculate Distance: House Point <-> Turbine Point
        # We use the preserved geometries
        candidates["dist_to_new_turbine"] = candidates[
            "house_geometry_original"
        ].distance(candidates["turb_geometry"])

        # 7. Select Nearest: Sort by distance and keep only the closest per house sale row
        # Grouping by the index of the original houses dataframe ensures we map correctly
        nearest_new_turbine = (
            candidates.sort_values("dist_to_new_turbine").groupby(level=0).head(1)
        )

        return nearest_new_turbine

    def _filter_active_turbines(self, turbines: gpd.GeoDataFrame):
        """Filter turbines to ensure we only consider those that were active in the relevant time window for each sale."""
        # 5. Temporal Filter
        # Turbine must be activated BETWEEN the previous sale and current sale
        # And optionally, ensure it wasn't decommissioned before the current sale
        mask_turbine_activated_after_prev_sale = (
            turbines["date_of_effect"] > turbines["salgs_dato_prev"]
        )
        mask_turbine_activated_before_current_sale = (
            turbines["date_of_effect"] <= turbines["salgs_dato"]
        )
        mask_turbine_not_decommissioned_before_current_sale = turbines[
            "afmelding_dato"
        ].isna() | (turbines["afmelding_dato"] > turbines["salgs_dato"])

        return turbines[
            (
                mask_turbine_activated_after_prev_sale
                & mask_turbine_activated_before_current_sale
                & mask_turbine_not_decommissioned_before_current_sale
            )
        ].copy()

    def _drop_cols(self, gdf: gpd.GeoDataFrame):
        """Drop columns that are not relevant for the ML model and could cause data leakage. This includes any columns related to the previous sale, as well as any columns that are not needed for the analysis."""
        return gdf.drop(
            columns=[
                "Kommunekode",
                "Kommunenavn",
                "Postnr",
                "Vejnavn",
                "HusNr",
                "PostDistrikt",
                "BenyttelseKode_T",
                "BenyttelseKode",
                "Valutakode",
                "byg406Koordinatsystem_T",
                "byg406Koordinatsystem",
                "KontantKoebesum",
                "Kommunekode_prev",
                "BenyttelseKode_prev",
                "Kommunenavn_prev",
                "Postnr_prev",
                "Vejnavn_prev",
                "HusNr_prev",
                "PostDistrikt_prev",
                "BenyttelseKode_T_prev",
                "BenyttelseKode_prev",
                "Valutakode_prev",
                "byg406Koordinatsystem_T_prev",
                "byg406Koordinatsystem_prev",
                "KontantKoebesum_prev",
                "byg021BygningensAnvendelse",
                "byg021BygningensAnvendelse_prev",
                "geometry_prev",
                "vurderingsaar_prev",
                "byg026OpFoerelsesAar_prev",
            ]
        )

    def _drop_rows_without_sale_price(self, gdf: gpd.GeoDataFrame):
        """Drop rows where the sale price is zero, as these do not represent actual sales and could skew the analysis."""
        _no_target = gdf[gdf["SamletKoebesum"] == 0]
        return gdf.drop(_no_target.index)


@app.cell
def _():
    return


@app.cell(column=6, hide_code=True)
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
