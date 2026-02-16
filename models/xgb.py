import os
from typing import Literal
import matplotlib
import matplotlib.pyplot as plt
from optuna_integration import XGBoostPruningCallback

matplotlib.use("Agg")
import functools
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import mlflow
import optuna
import chime
import shap

chime.theme("mario")  # Set a fun theme for notifications


def objective(
    trial: optuna.Trial,
    experiment_tracker: ExperimentTracker,
    data_handler: DataHandler,
    comparison_type: Literal["next", "all"],
    log_model: bool = False,
):

    # Start a new MLflow run for each trial
    with experiment_tracker.start_run(nested=True) as run:
        # Log Optuna trial parameters
        test_split = 0.3
        random_state = 42
        n_estimators = 10_000
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 6)
        subsample = trial.suggest_float("subsample", 0.6, 0.8)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        gamma = trial.suggest_float("gamma", 2, 5)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 1)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 10)

        mlflow.log_params(
            {
                "test_split": test_split,
                "random_state": random_state,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "gamma": gamma,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
                "trial_number": trial.number,
                "algorithm": experiment_tracker.algorithm,
                "comparison_type": experiment_tracker.comparison_type,
            }
        )

        try:
            X_train, X_test, y_train, y_test = data_handler.x_y_split(
                comparison_type=comparison_type,
                test_size=test_split,
                random_state=random_state,
            )
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=1,
                callbacks=[pruning_callback],
                early_stopping_rounds=50,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        except optuna.TrialPruned:
            # Delete the MLflow run for pruned trials to avoid clutter
            run_id = run.info.run_id
            mlflow.end_run()
            mlflow.delete_run(run_id)
            raise  # Re-raise to let Optuna handle the pruning
        except ValueError as e:
            mlflow.log_param("error", str(e))
            return float("inf")  # Return a large value to indicate failure
        mlflow.log_param("error", None)

        y_pred = model.predict(X_test)

        # Remove callbacks to avoid issues when cross-validating
        model.set_params(callbacks=[], early_stopping_rounds=None)

        # Calculate and log metrics
        neg_mse_cross_val = cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=10
        ).mean()
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metrics(
            {
                "neg_mse_cross_val": neg_mse_cross_val,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

        if log_model:
            mlflow.sklearn.log_model(model, "model")  # type: ignore

        # Log the model code as an artifact for reproducibility
        mlflow.log_artifact("models/xgb.py", artifact_path="model_code")

        # Log SHAP plots as artifacts
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False, max_display=None)
        shap_beeswarm_plot_path = f"shap_beeswarm_trial_{trial.number}.png"
        plt.savefig(shap_beeswarm_plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_beeswarm_plot_path, artifact_path="shap_plots")
        os.remove(shap_beeswarm_plot_path)  # Clean up the local file

        # SHAP summary plot for overall feature importance
        plt.figure()
        shap.plots.scatter(shap_values[:, "dist_to_new_turbine"], show=False)
        shap_summary_plot_path = f"shap_summary_scatter_trial_{trial.number}.png"
        plt.savefig(shap_summary_plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_summary_plot_path, artifact_path="shap_plots")
        os.remove(shap_summary_plot_path)  # Clean up the local file

        # SHAP bar plot for feature importance
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        shap_bar_plot_path = f"shap_bar_trial_{trial.number}.png"
        plt.savefig(shap_bar_plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_bar_plot_path, artifact_path="shap_plots")
        os.remove(shap_bar_plot_path)  # Clean up the local file

        return neg_mse_cross_val


if __name__ == "__main__":
    comparison_type: Literal["next", "all"] = "all"
    log_model = True
    exp_tracker = ExperimentTracker(
        algorithm="xgboost",
        comparison_type=comparison_type,
    )

    # Load the data
    data_handler = DataHandler()

    pruner = optuna.pruners.HyperbandPruner(max_resource=10_000)  # Matches n_estimators
    study = optuna.create_study(direction="maximize", pruner=pruner)
    objective_func = functools.partial(
        objective,
        experiment_tracker=exp_tracker,
        data_handler=data_handler,
        comparison_type=comparison_type,
        log_model=log_model,
    )

    study.optimize(
        objective_func,
        n_trials=300,
        show_progress_bar=True,
        n_jobs=-1,
    )

    if log_model:
        exp_tracker.find_and_register_best_model(
            experiment_name=exp_tracker.experiment_name,
            metric_name="neg_mse_cross_val",
            order_by="DESC",
        )

    chime.success()  # Notify when the script finishes
