from typing import Literal
import warnings
import logging
import mlflow


class ExperimentTracker:
    def __init__(
        self,
        algorithm: str,
        comparison_type: Literal["next", "all"],
        experiment_name: str = "Wind Turbine - House Price Prediction",
    ):
        self._suppress_logging()
        self.algorithm = algorithm
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        mlflow.autolog()
        mlflow.log_param("algorithm", self.algorithm)
        mlflow.log_param("dataset", comparison_type)

    def _suppress_logging(self):
        # Suppress MLflow warnings
        warnings.filterwarnings("ignore", module="mlflow")
        warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
        # Suppress MLflow logging
        logging.getLogger("mlflow").setLevel(logging.ERROR)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ):
        mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)

    def find_and_register_best_model(
        self,
        experiment_name: str,
        model_name: str | None = None,
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

        if model_name is None:
            model_name = f"best_model_{self.algorithm}"
        # Register the model
        mlflow.register_model(model_uri=model_uri, name=model_name)
