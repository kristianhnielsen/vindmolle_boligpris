import os
from typing import Literal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from optuna_integration import XGBoostPruningCallback

matplotlib.use("Agg")
import functools
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    n_estimators: int = 10_000,
):

    # Ensure MLflow state is initialized in this process (needed for n_jobs > 1)
    mlflow.set_experiment(experiment_tracker.experiment_name)
    mlflow.autolog(log_models=experiment_tracker.log_model)

    # Start a new MLflow run for each trial
    with experiment_tracker.start_run(nested=True) as run:
        # Log Optuna trial parameters

        test_split = 0.1
        # random_state = 42
        random_state = trial.suggest_int("random_state", 1, 10000)
        # months_of_effect = trial.suggest_int("months_of_effect", 6, 36, step=6)
        months_of_effect = data_handler.months_of_effect
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
        max_depth = trial.suggest_int("max_depth", 4, 7)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 9)
        subsample = trial.suggest_float("subsample", 0.6, 0.75)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        gamma = trial.suggest_float("gamma", 2, 7)
        reg_alpha = trial.suggest_float("reg_alpha", 0.5, 3)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 12)
        tree_method = "hist"

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
                "months_of_effect": months_of_effect,
                "tree_method": tree_method,
                "version": 50,
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
                enable_categorical=True,
                tree_method=tree_method,
            )

            weights = np.ones(len(y_train))
            # Weight treated samples higher (e.g., 10x or 100x depending on scarcity)
            weights[X_train["has_new_turbine"] == 1] = 100

            model.fit(
                X_train,
                y_train,
                sample_weight=weights,
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
            return float("inf")  # Return worst possible value for a maximization study
        mlflow.log_param("error", None)

        y_pred = model.predict(X_test)

        # Calculate and log metrics
        mse = mean_squared_error(y_test, y_pred)
        neg_mse = -mse
        rmse = mse**0.5
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_metrics(
            {
                "neg_mse": neg_mse,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        )

        if log_model:
            mlflow.sklearn.log_model(model, "model")  # type: ignore

        # Log the model code as an artifact for reproducibility
        mlflow.log_artifact(os.path.abspath(__file__), artifact_path="model_code")

        generate_shap_plots(model, X_test, trial.number)

        return neg_mse


def generate_shap_plots(model, X_test, trial_number):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=None)
    shap_beeswarm_plot_path = f"shap_beeswarm_trial_{trial_number}.png"
    plt.savefig(shap_beeswarm_plot_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(
        shap_beeswarm_plot_path, artifact_path="shap_plots"
    )  # Log the plot as an artifact
    os.remove(
        shap_beeswarm_plot_path
    )  # Remove the plot after logging it as an artifact

    plt.figure()
    shap.plots.scatter(shap_values[:, "dist_to_new_turbine"], show=False)
    shap_summary_plot_path = f"shap_scatter_trial_{trial_number}.png"
    plt.savefig(shap_summary_plot_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(
        shap_summary_plot_path, artifact_path="shap_plots"
    )  # Log the plot as an artifact
    os.remove(shap_summary_plot_path)  # Remove the plot after logging it as an artifact

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    shap_bar_plot_path = f"shap_bar_trial_{trial_number}.png"
    plt.savefig(shap_bar_plot_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(
        shap_bar_plot_path, artifact_path="shap_plots"
    )  # Log the plot as an artifact
    os.remove(shap_bar_plot_path)  # Remove the plot after logging it as an artifact

    # shap waterfall plot for the first X samples without turbine in the test set
    no_turbine_indices = np.where(X_test["has_new_turbine"] == 0)[0]
    has_new_turbine_indices = np.where(X_test["has_new_turbine"] == 1)[0]
    samples_to_plot = 5

    for i in np.random.choice(
        no_turbine_indices,
        size=min(samples_to_plot, len(no_turbine_indices)),
        replace=False,
    ):
        total_features = shap_values.shape[1]
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False, max_display=total_features)
        shap_waterfall_plot_path = (
            f"shap_waterfall_no_turbine_trial_{trial_number}_sample_{i}.png"
        )
        plt.savefig(shap_waterfall_plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(
            shap_waterfall_plot_path, artifact_path="shap_plots"
        )  # Log the plot as an artifact
        os.remove(
            shap_waterfall_plot_path
        )  # Remove the plot after logging it as an artifact

    # Plot waterfall for 4 random samples with turbine
    for i in np.random.choice(
        has_new_turbine_indices,
        size=min(samples_to_plot, len(has_new_turbine_indices)),
        replace=False,
    ):
        total_features = shap_values.shape[1]
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False, max_display=total_features)
        shap_waterfall_plot_path = (
            f"shap_waterfall_has_turbine_trial_{trial_number}_sample_{i}.png"
        )
        plt.savefig(shap_waterfall_plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(
            shap_waterfall_plot_path, artifact_path="shap_plots"
        )  # Log the plot as an artifact
        os.remove(
            shap_waterfall_plot_path
        )  # Remove the plot after logging it as an artifact


if __name__ == "__main__":
    comparison_type: Literal["next", "all"] = "next"
    log_model = False
    exp_tracker = ExperimentTracker(
        algorithm="xgboost",
        comparison_type=comparison_type,
    )

    # Load the data
    data_handler = DataHandler(months_of_effect=24)

    n_estimators = 20
    # n_estimators = 10  # Set n_estimators here to match the pruner's max_resource
    pruner = optuna.pruners.HyperbandPruner(max_resource=n_estimators)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    objective_func = functools.partial(
        objective,
        experiment_tracker=exp_tracker,
        data_handler=data_handler,
        comparison_type=comparison_type,
        log_model=log_model,
        n_estimators=n_estimators,
    )

    study.optimize(
        objective_func,
        n_trials=100,
        show_progress_bar=True,
    )

    if log_model:
        exp_tracker.find_and_register_best_model(
            experiment_name=exp_tracker.experiment_name,
            metric_name="neg_mse",
            order_by="DESC",
        )

    # chime.success()  # Notify when the script finishes
