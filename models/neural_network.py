import torch
import torch.nn as nn
import os
import pickle
from typing import Literal
import matplotlib
import matplotlib.pyplot as plt
from optuna_integration import PyTorchLightningPruningCallback

matplotlib.use("Agg")
import functools
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import mlflow
import optuna
import chime
import shap
import numpy as np


class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for regression."""

    def __init__(
        self, input_dim: int, hidden_layers: list[int], dropout_rate: float = 0.2
    ):
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def objective(
    trial: optuna.Trial,
    experiment_tracker: ExperimentTracker,
    data_handler: DataHandler,
    comparison_type: Literal["next", "all"],
    log_model: bool = False,
):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_layers = [
        trial.suggest_int(f"n_units_l{i}", 32, 512, log=True) for i in range(n_layers)
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    n_epochs = trial.suggest_int("n_epochs", 50, 300)

    # Get data
    X_train, X_test, y_train, y_test = data_handler.x_y_split(
        comparison_type=comparison_type, scale=True
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize model
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim, hidden_layers, dropout_rate)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Report intermediate value for pruning
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_test_tensor)
                val_mse = criterion(val_predictions, y_test_tensor).item()
            model.train()

            trial.report(val_mse, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Log to MLflow
    with experiment_tracker.start_run():
        mlflow.log_params(trial.params)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("neg_mse_cross_val", -test_mse)

        if log_model:
            # Save model state dict
            torch.save(model.state_dict(), "model.pth")
            mlflow.log_artifact("model.pth", artifact_path="model")

            # Save model as pickle
            model_path = "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, "model")
            os.remove(model_path)

    return test_mse


if __name__ == "__main__":
    comparison_type: Literal["next", "all"] = "all"
    log_model = False
    exp_tracker = ExperimentTracker(
        algorithm="neural_network",
        comparison_type=comparison_type,
    )

    # Load the data
    data_handler = DataHandler()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=10,  # Wait 10 epochs before pruning
    )
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
        n_trials=20,
        show_progress_bar=True,
        n_jobs=-1,
    )

    if log_model:
        exp_tracker.find_and_register_best_model(
            experiment_name=exp_tracker.experiment_name,
            metric_name="neg_mse_cross_val",
            order_by="DESC",
        )
