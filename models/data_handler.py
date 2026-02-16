from typing import Literal

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from preprocessing import get_comparative_sales_with_turbine


class DataHandler:

    def __init__(
        self,
        radius: int = 5000,
        months_of_effect: int = 24,
        target_variable: str = "price_diff",
    ):
        self.radius = radius
        self.months_of_effect = months_of_effect
        self.target_variable = target_variable

        self.data_next = self.get_data("next")
        self.data_all = self.get_data("all")

    def refresh_data(self):
        self.data_next = self.get_data("next")
        self.data_all = self.get_data("all")

    def get_data(self, comparison_type: Literal["next", "all"]):
        data = get_comparative_sales_with_turbine(
            on=comparison_type,
            radius_m=self.radius,
            months_of_effect=self.months_of_effect,
        )

        feature_engineered_data = self._feature_engineering(data)
        return feature_engineered_data

    def _feature_engineering(self, data):
        # data["days_since_assessment"] = (
        #     data["salgs_dato"] - data["vurderingsaar"]
        # ).dt.days  # type: ignore
        # data["grundvaerdi_diff"] = (
        #     data["GrundvaerdiBeloeb"] - data["GrundvaerdiBeloeb_prev"]
        # )
        # data["ejendomvaerdi_diff"] = (
        #     data["EjendomvaerdiBeloeb"] - data["EjendomvaerdiBeloeb_prev"]
        # )
        # data["koebesum_diff"] = data["SamletKoebesum"] - data["SamletKoebesum_prev"]
        data["vurderet_areal_diff"] = data["VURderetAreal"] - data["VURderetAreal_prev"]
        data["price_diff"] = data["SamletKoebesum"] - data["SamletKoebesum_prev"]

        data["turbine_height_m"] = data["rotor_diameter_m"] / 2 + data["height_pole_m"]

        data.drop(
            columns=[
                "rotor_diameter_m",
                "height_pole_m",
                "SamletKoebesum",
                "geometry",
                "salgs_dato",
                "salgs_dato_prev",
                "vurderingsaar",
                "byg038SamletBygningsAreal_prev",
                "byg039BygningensSamlBoligAreal_prev",
                "house_geometry_original",
                "tilslutning_dato",
                "date_of_effect",
                # "BFEnummer",
                "byg038SamletBygningsAreal",
                "byg039BygningensSamlBoligAreal",
                "GrundvaerdiBeloeb",
                "GrundvaerdiBeloeb_prev",
                "EjendomvaerdiBeloeb",
                "EjendomvaerdiBeloeb_prev",
                "SamletKoebesum_prev",
                "VURderetAreal",
                "VURderetAreal_prev",
                "growth_rate",
            ],
            inplace=True,
        )
        data.dropna(inplace=True)  # Drop rows with missing values

        # scaler = StandardScaler()
        # feature_cols = data.columns.difference([self.target_variable])
        # data[feature_cols] = scaler.fit_transform(data[feature_cols])

        return data

    def x_y_split(
        self,
        comparison_type: Literal["next", "all"] = "next",
        target=None,
        test_size=0.2,
        random_state=42,
    ):
        n_splits = 1

        if target is None:
            target = self.target_variable

        # Use GroupShuffleSplit to ensure same properties stay together
        gss = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )

        # Define features and target variable
        if comparison_type == "next":
            data = self.data_next
        else:
            data = self.data_all

        # Extract groups (BFEnummer) before dropping it from features
        groups = data["BFEnummer"]
        X = data.drop(columns=[target, "BFEnummer"])
        y = data[target]

        # Get train and test indices
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        return X_train, X_test, y_train, y_test
