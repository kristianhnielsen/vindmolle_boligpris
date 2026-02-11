import geopandas as gpd
from experiment_tracking import find_and_register_best_model
from preprocessing import get_comparative_sales_with_turbine
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import optuna
import warnings
import logging
import chime
import json
from datetime import datetime
from pathlib import Path

# Suppress MLflow warnings
warnings.filterwarnings("ignore", module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Suppress MLflow logging
logging.getLogger("mlflow").setLevel(logging.ERROR)

chime.theme("mario")  # Set a fun theme for notifications


def feature_engineering(data):

    # Prepare features and target variable

    data["days_since_assessment"] = (
        data["salgs_dato"] - data["vurderingsaar"]
    ).dt.days  # type: ignore
    data["grundvaerdi_diff"] = (
        data["GrundvaerdiBeloeb"] - data["GrundvaerdiBeloeb_prev"]
    )
    data["ejendomvaerdi_diff"] = (
        data["EjendomvaerdiBeloeb"] - data["EjendomvaerdiBeloeb_prev"]
    )
    # data["koebesum_diff"] = data["SamletKoebesum"] - data["SamletKoebesum_prev"]
    data["vurderet_areal_diff"] = data["VURderetAreal"] - data["VURderetAreal_prev"]

    data.drop(
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
            "growth_rate",
        ],
        inplace=True,
    )
    data.dropna(inplace=True)  # Drop rows with missing values

    scaler = StandardScaler()
    feature_cols = data.columns.difference(["SamletKoebesum"])
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    return data


def objective(trial: optuna.Trial, data: gpd.GeoDataFrame):

    # Start a new MLflow run for each trial
    with mlflow.start_run(nested=True):
        # Log Optuna trial parameters
        test_split = 0.1
        random_state = trial.suggest_int("random_state", 0, 5)
        tolerance = trial.suggest_float("tolerance", 0.00001, 0.1)
        solver = trial.suggest_categorical(
            "solver",
            [
                "svd",
                "cholesky",
                "lsqr",
                "sparse_cg",
                "sag",
                "saga",
                "lbfgs",
            ],
        )
        alpha = trial.suggest_float("alpha", 0.1, 10.0)
        solver_ignores_tol = solver in ["svd", "cholesky"]

        mlflow.log_params(
            {
                "tolerance": tolerance,
                "solver": solver,
                "alpha": alpha,
                "test_split": test_split,
                "random_state": random_state,
                "solver_ignores_tol": solver_ignores_tol,
                "trial_number": trial.number,
            }
        )

        try:
            X, y = x_y_split(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state
            )

            model = Ridge(tol=tolerance, solver=solver, alpha=alpha)  # type: ignore
            model.fit(X_train, y_train)
        except ValueError as e:
            mlflow.log_param("error", str(e))
            return float("inf")  # Return a large value to indicate failure
        mlflow.log_param("error", None)

        y_pred = model.predict(X_test)

        # Calculate and log metrics
        neg_mse_cross_val = cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        ).mean()
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])

        mlflow.log_metrics(
            {
                "neg_mse_cross_val": neg_mse_cross_val,
                "rmse": rmse,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
            }
        )

        # Log model
        mlflow.sklearn.log_model(model, "model")  # type: ignore

        mlflow.log_artifact("models/linear_regression.py", artifact_path="model_code")

        return neg_mse_cross_val


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    return adjusted_r2


def x_y_split(data, target="SamletKoebesum"):
    # Define features and target variable
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


if __name__ == "__main__":
    comparison_type = "next"
    experiment_name = f"Ridge Regression - {comparison_type.capitalize()} Comparisons"
    mlflow.set_experiment(experiment_name)
    mlflow.set_experiment_tags(
        {"model_type": "ridge_regression", "comparison_type": comparison_type}
    )
    mlflow.sklearn.autolog()  # type: ignore

    # Load the data
    data = get_comparative_sales_with_turbine(on=comparison_type)
    data = feature_engineering(data)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, data),
        n_trials=1000,
        show_progress_bar=True,
        n_jobs=-1,
    )

    find_and_register_best_model(
        experiment_name=experiment_name,
        model_name=f"ridge_regression_{comparison_type}_model",
    )

    chime.success()  # Notify when the script finishes
