import warnings
import keras

import os
from typing import Literal
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from experiment_tracking import ExperimentTracker
from data_handler import DataHandler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import chime
import shap
import numpy as np

warnings.filterwarnings("ignore", module="absl")


config = {
    "log_model": False,
    "run_name": None,
    "EPOCHS": 1_000,
    "BATCH_SIZE": [64],
    "ES_PATIENCE": 20,
    "REDUCE_LR_PATIENCE": 5,
    "REDUCE_LR_FACTOR": [0.5],
    "LAYERS": [1, 3, 4, 6],
    "THICKNESS": [1, 2],
    "LAST_LAYER_EXPONENT": [6],
    "DROPOUT_RATE": [0.0],
    "BATCH_NORM": [True],
    "INIT_LR": [0.001],
    # "L1": [0.0, 0.001, 0.01, 0.1, 0.3],
    # "L2": [0.01, 0.001, 0.01, 0.1, 0.3],
    "L1": [0.0],
    "L2": [0.0],
    "ACTIVATION": ["tanh", "leaky_relu", "relu"],
    "VERSION": 1,
}


def run_model(config: dict):
    comparison_type: Literal["next", "all"] = "all"

    optimizer = keras.optimizers.Adam(learning_rate=config["INIT_LR"])
    regularizer = keras.regularizers.L1L2(config["L1"], config["L2"])

    exp_tracker = ExperimentTracker(
        algorithm="neural_network_tf",
        comparison_type=comparison_type,
        log_model=config["log_model"],
    )

    # Load the data
    data_handler = DataHandler()

    X_train, X_test, y_train, y_test = data_handler.x_y_split(
        comparison_type=comparison_type, scale=True
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config["ES_PATIENCE"], restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config["REDUCE_LR_FACTOR"],
        patience=config["REDUCE_LR_PATIENCE"],
    )

    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    for i in range(config["LAYERS"]):
        exp = 2 ** (
            config["LAYERS"] - i + config["LAST_LAYER_EXPONENT"]
        )  # Exponential decay of layer sizes
        for _ in range(config["THICKNESS"]):
            model.add(
                keras.layers.Dense(
                    exp, activation=config["ACTIVATION"], kernel_regularizer=regularizer
                )
            )
        model.add(
            keras.layers.Dense(
                exp, activation=config["ACTIVATION"], kernel_regularizer=regularizer
            )
        )

        model.add(keras.layers.Dropout(config["DROPOUT_RATE"]))
        if config["BATCH_NORM"]:
            model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1))

    with exp_tracker.start_run(run_name=config["run_name"]) as run:
        model.compile(
            optimizer=optimizer,  # type: ignore
            loss="mse",
            metrics=["mae"],
        )
        mlflow.log_params(
            {
                "algorithm": exp_tracker.algorithm,
                "comparison_type": exp_tracker.comparison_type,
                "epochs": config["EPOCHS"],
                "batch_size": config["BATCH_SIZE"],
                "early_stopping_patience": config["ES_PATIENCE"],
                "reduce_lr_patience": config["REDUCE_LR_PATIENCE"],
                "reduce_lr_factor": config["REDUCE_LR_FACTOR"],
                "layers": config["LAYERS"],
                "last_layer_exponent": config["LAST_LAYER_EXPONENT"],
                "dropout_rate": config["DROPOUT_RATE"],
                "batch_norm": config["BATCH_NORM"],
                "init_lr": config["INIT_LR"],
                "l1": config["L1"],
                "l2": config["L2"],
                "activation": config["ACTIVATION"],
                "thickness": config["THICKNESS"],
                "grid_search": True,
                "version": config["VERSION"],
            }
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=config["EPOCHS"],
            validation_data=(X_test, y_test),
            batch_size=config["BATCH_SIZE"],
            callbacks=[early_stopping, reduce_lr],
        )

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        mae = mean_absolute_error(y_test, y_pred)

        exp_tracker.log_metrics(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
            }
        )

        mlflow.log_artifact("models/tf_neural_network.py", artifact_path="model_code")

        # SHAP analysis

        # Get feature names from training data
        feature_names = X_train.columns.tolist()

        # Convert to numpy arrays for SHAP
        X_train_array = X_train.values
        X_test_array = X_test.values

        # Sample a subset of training data for background (200 samples is typical)
        background_size = min(200, len(X_train_array))
        background_indices = np.random.choice(
            len(X_train_array), background_size, replace=False
        )
        background = X_train_array[background_indices]

        # Use GradientExplainer for better TensorFlow 2.x compatibility
        # GradientExplainer works better with BatchNormalization and modern TF
        explainer = shap.GradientExplainer(model, background)

        # Calculate SHAP values for test set (use subset if too large)
        test_sample_size = min(100, len(X_test_array))
        X_test_sample = X_test_array[:test_sample_size]
        shap_values = explainer.shap_values(X_test_sample)

        # Squeeze to remove extra dimensions if needed
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For single output models
        shap_values = np.squeeze(shap_values)

        # For GradientExplainer, compute expected value from background predictions
        background_predictions = model.predict(background)
        expected_value = float(np.mean(background_predictions))

        # Create directory for SHAP plots
        shap_dir = "shap_plots"
        os.makedirs(shap_dir, exist_ok=True)

        # 1. Beeswarm plot
        plt.figure()
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values,
                base_values=np.full(len(shap_values), expected_value),
                data=X_test_sample,
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        summary_plot_path = os.path.join(shap_dir, "shap_beeswarm_plot.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(summary_plot_path, artifact_path="shap_plots")

        # 2. Bar plot
        plt.figure()
        shap.plots.bar(
            shap.Explanation(
                values=shap_values,
                base_values=np.full(len(shap_values), expected_value),
                data=X_test_sample,
                feature_names=feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        bar_plot_path = os.path.join(shap_dir, "shap_bar_plot.png")
        plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(bar_plot_path, artifact_path="shap_plots")

        for i in range(min(7, len(X_test_sample))):
            # 3. Waterfall plot for each of the first 3 predictions
            plt.figure()
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[i],
                    base_values=expected_value,
                    data=X_test_sample[i],
                    feature_names=feature_names,
                ),
                show=False,
            )
            plt.tight_layout()
            waterfall_plot_path = os.path.join(
                shap_dir, f"shap_waterfall_plot_sample_{i}.png"
            )
            plt.savefig(waterfall_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(waterfall_plot_path, artifact_path="shap_plots")

            # 4. Force plot for first prediction (saved as HTML)
            # shap.initjs()
            force_plot = shap.force_plot(
                expected_value,
                shap_values[i],
                X_test_sample[i],
                feature_names=feature_names,
            )
            force_plot_path = os.path.join(shap_dir, f"shap_force_plot_sample_{i}.html")
            shap.save_html(force_plot_path, force_plot)
            mlflow.log_artifact(force_plot_path, artifact_path="shap_plots")

    if config["log_model"]:
        exp_tracker.find_and_register_best_model(
            experiment_name=exp_tracker.experiment_name,
            metric_name="neg_mse_cross_val",
            order_by="DESC",
        )


if __name__ == "__main__":
    comparison_type: Literal["next", "all"] = "next"

    # comfig combinations
    config_combinations = []
    for batch_size in config["BATCH_SIZE"]:
        for reduce_lr_factor in config["REDUCE_LR_FACTOR"]:
            for layers in config["LAYERS"]:
                for last_layer_exponent in config["LAST_LAYER_EXPONENT"]:
                    for dropout_rate in config["DROPOUT_RATE"]:
                        for batch_norm in config["BATCH_NORM"]:
                            for init_lr in config["INIT_LR"]:
                                for l1 in config["L1"]:
                                    for l2 in config["L2"]:
                                        for activation in config["ACTIVATION"]:
                                            for thickness in config["THICKNESS"]:
                                                config_combinations.append(
                                                    {
                                                        "log_model": config[
                                                            "log_model"
                                                        ],
                                                        "run_name": config["run_name"],
                                                        "BATCH_SIZE": batch_size,
                                                        "EPOCHS": config["EPOCHS"],
                                                        "THICKNESS": thickness,
                                                        "ES_PATIENCE": config[
                                                            "ES_PATIENCE"
                                                        ],
                                                        "REDUCE_LR_PATIENCE": config[
                                                            "REDUCE_LR_PATIENCE"
                                                        ],
                                                        "REDUCE_LR_FACTOR": reduce_lr_factor,
                                                        "LAYERS": layers,
                                                        "LAST_LAYER_EXPONENT": last_layer_exponent,
                                                        "DROPOUT_RATE": dropout_rate,
                                                        "BATCH_NORM": batch_norm,
                                                        "INIT_LR": init_lr,
                                                        "L1": l1,
                                                        "L2": l2,
                                                        "ACTIVATION": activation,
                                                        "VERSION": config["VERSION"],
                                                    }
                                                )
    print(f"Total combinations to run: {len(config_combinations)}")
    input("Press Enter to start running the models...")
    # Run models for all combinations
    chime.theme("mario")
    for i, config_combination in enumerate(config_combinations[:1]):
        print(
            f"\n\n\nRunning combination {i+1}/{len(config_combinations)}    #############################\n\n\n"
        )
        chime.info()
        run_model(config_combination)

    chime.success()
