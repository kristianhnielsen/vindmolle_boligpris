import functools
import os
from typing import Literal
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import mlflow
import optuna
import chime
import matplotlib.pyplot as plt


chime.theme("mario")  # Set a fun theme for notifications


def objective(trial: optuna.Trial, data_handler: DataHandler, exp_tracker: ExperimentTracker) -> float:

    # Start a new MLflow run for each trial
    with mlflow.start_run(nested=True) as run:
        test_split = 0.1
        random_state = 42
        # random_state = trial.suggest_int("random_state", 1, 10000)

        # Log Optuna trial parameters
        # tolerance = trial.suggest_float("tolerance", 0.00001, 0.1)
        # solver = trial.suggest_categorical(
        #     "solver",
        #     [
        #         "svd",
        #         "cholesky",
        #         "lsqr",
        #         "sparse_cg",
        #         "sag",
        #         "saga",
        #         "lbfgs",
        #     ],
        # )
        # solver = 'auto'
        # alpha = trial.suggest_float("alpha", 1e-4, 1e3)
        # solver_ignores_tol = solver in ["svd", "cholesky", 'sparse_cg']
        # positive = trial.suggest_categorical("positive", [True, False])
        # max_iter = trial.suggest_int("max_iter", 50, 10000, step=50)
        max_iter = None
        mlflow.log_params(
            {
                "comparison_type": exp_tracker.comparison_type,
                'algorithm': exp_tracker.algorithm,
                'test_split': test_split,
                'random_state': random_state,
                # "tolerance": tolerance,
                # "solver": solver,
                # "alpha": alpha,
                'max_iter': max_iter,
                # "solver_ignores_tol": solver_ignores_tol,
                "trial_number": trial.number,
                # "positive": positive,
                'version': 24
            }
        )
        X_train, X_test, y_train, y_test = data_handler.x_y_split(
                comparison_type=exp_tracker.comparison_type,
                test_size=test_split,
                random_state=random_state,
            )

        try:
            model = Ridge(solver=solver, positive=positive)  # type: ignore
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
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        adjusted_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])

        mlflow.log_metrics(
            {
                "neg_mse_cross_val": neg_mse_cross_val,
                "mae": mae,
                "rmse": rmse,
                "mse": mse,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
            }
        )
        # Coefficients by name
        coef_dict = {f"coef_{col}": coef for col, coef in zip(X_train.columns, model.coef_)}
        mlflow.log_params(coef_dict)
        
        # plot coefficients
        plt.figure(figsize=(10, 6))
        plt.bar(X_train.columns, model.coef_)
        plt.xticks(rotation=90)
        plt.title("Feature Coefficients")
        plt.tight_layout()
        plt.savefig("coef_plot.png")
        mlflow.log_artifact("coef_plot.png", artifact_path="plots")
        os.remove("coef_plot.png")
        
        
        # Log model
        mlflow.sklearn.log_model(model, "model")  # type: ignore

        mlflow.log_artifact("models/linear_regression.py", artifact_path="model_code")

        return -neg_mse_cross_val


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    return adjusted_r2


if __name__ == "__main__":
    comparison_type: Literal["next", "all"] = "next"
    exp_tracker = ExperimentTracker(
        algorithm="ridge_regression",
        comparison_type=comparison_type,
    )

    # Load the data
    data_handler = DataHandler()
    
    study = optuna.create_study(direction="minimize")
    objective_func = functools.partial(
        objective,
        data_handler=data_handler,
        exp_tracker=exp_tracker,
    )
    
    study.optimize(
        objective_func,
        n_trials=5,
        show_progress_bar=True,
        n_jobs=1,
    )

    # exp_tracker.find_and_register_best_model(
    #     experiment_name=exp_tracker.experiment_name
    # )

    # chime.success()  # Notify when the script finishes
