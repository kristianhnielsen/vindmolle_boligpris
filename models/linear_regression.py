from preprocessing import get_comparative_sales_with_turbine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
from mlflow.models import infer_signature, evaluate
from mlflow.sklearn import log_model
import shap
import optuna
from optuna.integration.mlflow import MLflowCallback
import warnings

# Suppress MLflow schema inference warnings
warnings.filterwarnings("ignore", module="mlflow")


mlflow.sklearn.autolog()  # type: ignore # Enable automatic logging of parameters, metrics, and models
mlflow_callback = MLflowCallback(
    tracking_uri="sqlite:///mlflow.db",  # Set your MLflow tracking URI
    metric_name="rmse",  # Name of the metric to log
    create_experiment=True,  # Create a new experiment if it doesn't exist
)


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
    return data


@mlflow_callback.track_in_mlflow()  # Decorator to track the objective function in MLflow
def objective(trial: optuna.Trial):
    # Hyperparameter optimization logic here
    comparison_type = "all"
    test_split = 0.2
    random_state = 42
    mlflow.set_experiment_tags(
        {"model_type": "linear_regression", "comparison_type": comparison_type}
    )
    mlflow.sklearn.autolog()  # type: ignore # Enable automatic logging of parameters, metrics, and models

    # Load the data
    data = get_comparative_sales_with_turbine(on=comparison_type)

    data = feature_engineering(data)
    X, y = x_y_split(data)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_state
    )

    tolerance: float = trial.suggest_float(
        "tolerance",
        1e-6,
        1e-2,
        log=True,
    )

    model = LinearRegression(n_jobs=-1, tol=tolerance)  # type: ignore

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])

    mlflow.log_metric("rmse", rmse)  # Log RMSE to MLflow
    mlflow.log_metric("r2", r2)  # Log R² score to MLflow
    mlflow.log_metric("adjusted_r2", adjusted_r2)

    mlflow.end_run()
    return adjusted_r2  # An objective value linked with the Trial object.


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

    study = optuna.create_study(study_name="LinearRegression_all", direction="maximize")
    study.optimize(
        lambda trial: objective(trial),
        n_trials=10,
        callbacks=[mlflow_callback],
        show_progress_bar=True,
    )
