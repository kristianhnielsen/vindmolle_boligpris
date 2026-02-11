import mlflow


def find_and_register_best_model(
    experiment_name: str,
    model_name: str,
    filter_string: str = "",
    metric_name="neg_mse_cross_val",
    order_by="DESC",  # DESC for metrics to maximize, ASC for metrics to minimize
):
    """
    Find the best model across all experiments and register it.

    Parameters:
    -----------
    metric_name : str
        The metric to use for finding the best model (e.g., 'neg_mse_cross_val', 'r2', 'rmse')
    model_name : str
        The name to register the model under in MLflow Model Registry
    order_by : str
        'DESC' for maximizing metrics (e.g., r2, neg_mse_cross_val)
        'ASC' for minimizing metrics (e.g., rmse, mse)
    """

    # Search for all runs across all experiments
    # Order by the metric to get the best one
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        search_all_experiments=True,
        filter_string=filter_string,
        order_by=[f"metrics.{metric_name} {order_by}"],
        max_results=1,
        output_format="list",
    )

    best_run = runs[0]

    model_uri = f"runs:/{best_run.info.run_id}/model"  # type: ignore

    # Register the model
    mlflow.register_model(model_uri=model_uri, name=model_name)
