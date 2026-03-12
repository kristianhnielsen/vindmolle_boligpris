import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")

with app.setup(hide_code=True):
    import marimo as mo
    from scipy import stats
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        Lars,
        LassoLars,
        ElasticNet,
        QuantileRegressor,
        TweedieRegressor,
        HuberRegressor,
        RANSACRegressor,
        TheilSenRegressor,
        GammaRegressor,
        SGDRegressor,
        BayesianRidge,
        ARDRegression,
        OrthogonalMatchingPursuit,
    )
    from sklearn.svm import LinearSVR
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
    import optuna
    import mlflow
    import functools


@app.cell
def _():
    from models.data_handler import DataHandler

    return (DataHandler,)


@app.cell
def _(DataHandler):
    _data = DataHandler().get_data("next")
    _data
    return


@app.cell
def _():
    # models = {
    #     "Linear Regression": LinearRegression(),
    #     "Ridge Regression": Ridge(),
    #     "Lasso Regression": Lasso(),
    #     "LARS": Lars(),
    #     "LassoLARS": LassoLars(),
    #     "Elastic Net": ElasticNet(),
    #     "Quantile Regression": QuantileRegressor(),
    #     "Tweedie Regressor": TweedieRegressor(),
    #     "Huber Regressor": HuberRegressor(),
    #     "RANSAC Regressor": RANSACRegressor(),
    #     "Theil-Sen Regressor": TheilSenRegressor(),
    #     "Gamma Regressor": GammaRegressor(),
    #     "SGD Regressor": SGDRegressor(),
    #     "Bayesian Ridge": BayesianRidge(),
    #     "ARD Regression": ARDRegression(),
    #     "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit(),
    #     "Linear SVR": LinearSVR(),
    #     "poly_lasso": Pipeline(
    #         [
    #             (
    #                 "poly",
    #                 PolynomialFeatures(
    #                     degree=2,
    #                     interaction_only=False,  # set True for ONLY interactions
    #                     include_bias=False,
    #                 ),
    #             ),
    #             ("lasso", Lasso(alpha=0.001, max_iter=5000)),
    #         ]
    #     ),
    #     "spline_ridge": Pipeline(
    #         [
    #             (
    #                 "splines",
    #                 SplineTransformer(n_knots=6, degree=3, include_bias=False),
    #             ),
    #             ("scaler", StandardScaler()),
    #             ("ridge", Ridge(alpha=1.0)),
    #         ]
    #     ),
    # }
    return


@app.cell
def _():
    # comparison_types = "next"
    # random_state = 42
    # target_vars = [
    #     "SamletKoebesum",
    #     "growth_rate",
    #     "price_change",
    #     # "dist_to_new_turbine",
    # ]

    # results = {}
    # test_size = 0.2
    # for target_variable in target_vars:
    #     print(f"\n\n\nProcessing target variable: {target_variable}")

    #     X_train, X_test, y_train, y_test = DataHandler().x_y_split(
    #         comparison_type="next",
    #         target=target_variable,
    #         random_state=random_state,
    #         test_size=test_size,
    #         scale=True,
    #     )

    #     for name, model in models.items():
    #         try:
    #             print(f"Training {name}...")
    #             model.fit(X_train, y_train)
    #             y_pred = model.predict(X_test)
    #             mse = mean_squared_error(y_test, y_pred)
    #             r2 = r2_score(y_test, y_pred)
    #             mae = mean_absolute_error(y_test, y_pred)
    #             results[f"{name}_{target_variable}"] = {
    #                 "MSE": mse,
    #                 "R2": r2,
    #                 "MAE": mae,
    #                 "Target Variable": target_variable,
    #                 "Error": None,
    #             }
    #         except Exception as e:
    #             results[f"{name}_{target_variable}"] = {
    #                 "MSE": None,
    #                 "R2": None,
    #                 "MAE": None,
    #                 "Target Variable": target_variable,
    #                 "Error": e,
    #             }
    return


@app.cell
def _():
    # results_df = pd.DataFrame(results).T.reset_index()
    # results_df
    return


@app.cell
def _():
    # results_df.to_csv('model_selection_results2.csv')
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    # DiD Analysis
    """)
    return


@app.cell
def _(DataHandler):
    data = DataHandler().get_data("next")
    return (data,)


@app.cell(hide_code=True)
def _(data):
    if hasattr(data, "geometry"):
        data.drop(columns=["geometry"], inplace=True)

    # Clean data
    data.dropna(
        subset=[
            "price_change",
            "has_new_turbine",
            "dist_to_new_turbine",
            "visual_prominence",
            "SamletKoebesum_prev",
            "years_diff",
        ],
        inplace=True,
    )
    return


@app.cell(hide_code=True)
def _(data):
    print(f"Dataset size: {len(data)} property sales pairs")
    print(
        f"  Treatment group (near turbine): {(data['has_new_turbine'] == 1).sum()} ({100 * (data['has_new_turbine'] == 1).sum() / len(data):.1f}%)"
    )
    print(
        f"  Control group (no turbine):     {(data['has_new_turbine'] == 0).sum()} ({100 * (data['has_new_turbine'] == 0).sum() / len(data):.1f}%)"
    )
    return


@app.cell(hide_code=True)
def _(data):
    # Prepare features
    X_cols = [
        "has_new_turbine",
        "dist_to_new_turbine",
        "visual_prominence",
        "SamletKoebesum_prev",
        "years_diff",
        "dist_to_center",
    ]

    X = data[X_cols].values
    y = data["price_change"].values
    X_scaled = StandardScaler().fit_transform(X)
    return X_cols, X_scaled, y


@app.cell
def _():
    models = {
        "Linear Regression": LinearRegression,
        "Elastic Net": ElasticNet,
        "LARS": Lars,
        "LassoLARS": LassoLars,
        # "Quantile Regression": QuantileRegressor,
        # "Tweedie Regressor": TweedieRegressor,
        # "Huber Regressor": HuberRegressor,
        # "Theil-Sen Regressor": TheilSenRegressor,
        "SGD Regressor": SGDRegressor,
        "Bayesian Ridge": BayesianRidge,
        # "ARD Regression": ARDRegression,
        # "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit,
        # "Linear SVR": LinearSVR,
    }
    return (models,)


@app.cell(hide_code=True)
def _():
    from models.experiment_tracking import ExperimentTracker
    from models.visualization import (
        create_predictions_plot,
        create_residuals_plot,
        create_residuals_distribution,
        create_coefficients_plot,
        create_metrics_summary,
    )

    return (
        ExperimentTracker,
        create_coefficients_plot,
        create_metrics_summary,
        create_predictions_plot,
        create_residuals_distribution,
        create_residuals_plot,
    )


@app.cell
def _(
    ExperimentTracker,
    X_cols,
    X_scaled,
    create_coefficients_plot,
    create_metrics_summary,
    create_predictions_plot,
    create_residuals_distribution,
    create_residuals_plot,
    y,
):
    def objective(
        trial: optuna.Trial,
        experiment_tracker: ExperimentTracker,
        model,
    ):

        # Ensure MLflow state is initialized in this process (needed for n_jobs > 1)
        mlflow.set_experiment(experiment_tracker.experiment_name)
        mlflow.autolog(log_models=experiment_tracker.log_model)

        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "trial": trial.number,
                    "version": 13,
                    "algorithm": experiment_tracker.algorithm,
                }
            )

            model_name = experiment_tracker.algorithm
            params = {}

            match model_name:
                case "Linear Regression":
                    params["tol"] = trial.suggest_float("tol", 1e-8, 1e-4)
                    params["positive"] = trial.suggest_categorical(
                        "positive", [True, False]
                    )

                case "Elastic Net":
                    params["alpha"] = trial.suggest_float("alpha", 0.001, 1.0)
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
                    params["tol"] = trial.suggest_float("tol", 1e-8, 1e-4)
                    params["positive"] = trial.suggest_categorical(
                        "positive", [True, False]
                    )

                case "LARS":
                    params["n_nonzero_coefs"] = trial.suggest_int(
                        "n_nonzero_coefs", 100, 1000, step=100
                    )

                case "LassoLARS":
                    params["alpha"] = trial.suggest_float("alpha", 0.01, 1.0)
                    params["positive"] = trial.suggest_categorical(
                        "positive", [True, False]
                    )

                case "Quantile Regression":
                    params["alpha"] = trial.suggest_float("alpha", 0.001, 1.0)
                    params["solver"] = trial.suggest_categorical(
                        "solver", ["highs", "highs-ds", "highs-ipm"]
                    )

                case "Tweedie Regressor":
                    params["alpha"] = trial.suggest_float("alpha", 0.001, 1.0)
                    params["tol"] = trial.suggest_float("tol", 1e-8, 1e-4)

                case "Huber Regressor":
                    params["alpha"] = trial.suggest_float("alpha", 0.0001, 0.1)
                    params["epsilon"] = trial.suggest_float("epsilon", 1.0, 2.0)
                    params["tol"] = trial.suggest_float("tol", 1e-8, 1e-4)

                case "Theil-Sen Regressor":
                    params["tol"] = trial.suggest_float("tol", 1e-4, 1e-2)

                case "SGD Regressor":
                    params["alpha"] = trial.suggest_float("alpha", 0.00001, 0.01)
                    params["tol"] = trial.suggest_float("tol", 1e-4, 1e-2)
                    params["loss"] = trial.suggest_categorical(
                        "loss", ["squared_error", "huber", "epsilon_insensitive"]
                    )

                case "Bayesian Ridge":
                    params["max_iter"] = trial.suggest_int("max_iter", 100, 500)
                    params["alpha_1"] = trial.suggest_float("alpha_1", 1e-7, 1e-5)
                    params["alpha_2"] = trial.suggest_float("alpha_2", 1e-7, 1e-5)

                case "ARD Regression":
                    params["max_iter"] = trial.suggest_int("max_iter", 100, 500)
                    params["alpha_1"] = trial.suggest_float("alpha_1", 1e-7, 1e-5)
                    params["alpha_2"] = trial.suggest_float("alpha_2", 1e-7, 1e-5)

                case "Orthogonal Matching Pursuit":
                    params["n_nonzero_coefs"] = trial.suggest_int(
                        "n_nonzero_coefs", 1, 5
                    )

                case "Linear SVR":
                    params["C"] = trial.suggest_float("C", 0.1, 10.0)
                    params["epsilon"] = trial.suggest_float("epsilon", 0.01, 1.0)
                    params["tol"] = trial.suggest_float("tol", 1e-4, 1e-2)

                case _:
                    pass

            model = model(**params)

            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            # Calculate statistics
            n = len(y)
            k = X_scaled.shape[1]
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

            residuals = y - y_pred
            residual_std_error = np.sqrt(np.sum(residuals**2) / (n - k - 1))
            var_covar_matrix = residual_std_error**2 * np.linalg.inv(
                X_scaled.T @ X_scaled
            )
            std_errors = np.sqrt(np.diag(var_covar_matrix))
            t_stats = model.coef_ / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

            mlflow.log_metrics(
                {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "adjr2": adj_r2,
                }
            )

            for _col in X_cols:
                _col_metrics = {}
                turbine_idx = X_cols.index(_col)
                _col_metrics[f"coef_{_col}"] = model.coef_[turbine_idx]
                _col_metrics[f"std_e_{_col}"] = std_errors[turbine_idx]
                _col_metrics[f"p_{_col}"] = p_values[turbine_idx]
                mlflow.log_metrics(_col_metrics)

            # Create and log visualizations
            try:
                # Actual vs Predicted plot
                pred_plot_path = create_predictions_plot(
                    y, y_pred, experiment_tracker.algorithm
                )
                mlflow.log_artifact(pred_plot_path, "plots")

                # Residuals plot
                residuals_plot_path = create_residuals_plot(
                    y, y_pred, experiment_tracker.algorithm
                )
                mlflow.log_artifact(residuals_plot_path, "plots")

                # Residuals distribution
                dist_plot_path = create_residuals_distribution(
                    y, y_pred, experiment_tracker.algorithm
                )
                mlflow.log_artifact(dist_plot_path, "plots")

                # Feature coefficients
                coef_dict = {col: coef for col, coef in zip(X_cols, model.coef_)}
                coef_plot_path = create_coefficients_plot(
                    coef_dict, experiment_tracker.algorithm
                )
                mlflow.log_artifact(coef_plot_path, "plots")

                # Metrics summary
                summary_plot_path = create_metrics_summary(
                    y, y_pred, experiment_tracker.algorithm
                )
                mlflow.log_artifact(summary_plot_path, "plots")

            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")

            return rmse

    return (objective,)


@app.cell
def _(ExperimentTracker, models, objective):
    for name, model in models.items():
        print(f"Training {name}...")
        exp_tracker = ExperimentTracker(
            algorithm=name,
            comparison_type="next",
        )

        study = optuna.create_study(direction="minimize")
        objective_func = functools.partial(
            objective, experiment_tracker=exp_tracker, model=model
        )

        study.optimize(
            objective_func,
            n_trials=20,
            show_progress_bar=True,
        )
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
