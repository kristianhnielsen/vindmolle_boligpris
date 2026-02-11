import geopandas as gpd
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from preprocessing import get_comparative_sales_with_turbine
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import mlflow
import optuna
import chime

chime.theme("mario")  # Set a fun theme for notifications


def objective(trial: optuna.Trial, X, y):

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
        positive = trial.suggest_categorical("positive", [True, False])

        mlflow.log_params(
            {
                "tolerance": tolerance,
                "solver": solver,
                "alpha": alpha,
                "test_split": test_split,
                "random_state": random_state,
                "solver_ignores_tol": solver_ignores_tol,
                "trial_number": trial.number,
                "positive": positive,
            }
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_state
            )

            model = Ridge(tol=tolerance, solver=solver, alpha=alpha, positive=positive)  # type: ignore
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


if __name__ == "__main__":
    comparison_type = "next"
    exp_tracker = ExperimentTracker(
        algorithm="ridge_regression",
        comparison_type=comparison_type,
    )

    # Load the data
    data_handler = DataHandler()
    X, y = data_handler.x_y_split(comparison_type=comparison_type)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
        ),
        n_trials=1000,
        show_progress_bar=True,
        n_jobs=-1,
    )

    exp_tracker.find_and_register_best_model(
        experiment_name=exp_tracker.experiment_name
    )

    chime.success()  # Notify when the script finishes
