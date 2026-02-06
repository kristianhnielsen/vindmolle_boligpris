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


# Configuration
comparison_type = "all"
test_split = 0.1
random_state = 42
experiment_name = f"linear_regression_{comparison_type}"

mlflow.set_registry_uri("sqlite:///mlflow.db")  # Set your MLflow registry URI
mlflow.set_experiment_tags(
    {"model_type": "linear_regression", "comparison_type": comparison_type}
)
mlflow.sklearn.autolog()  # type: ignore # Enable automatic logging of parameters, metrics, and models
mlflow.set_experiment(experiment_name)

# Load the data
data = get_comparative_sales_with_turbine(on=comparison_type)
print(f"Data loaded with {len(data)} records for comparison type '{comparison_type}'.")

# Prepare features and target variable

data["days_since_assessment"] = (
    data["salgs_dato"] - data["vurderingsaar"]
).dt.days  # type: ignore
data["grundvaerdi_diff"] = data["GrundvaerdiBeloeb"] - data["GrundvaerdiBeloeb_prev"]
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
# Define features and target variable
target = "SamletKoebesum"
X = data.drop(columns=[target])
y = data[target]

# Feature engineering
# scaler = StandardScaler()
# numerical_features = [
#     col for col in X.columns if data[col].dtype in ["int64", "float64"]
# ]
# X[numerical_features] = scaler.fit_transform(X[numerical_features])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_split, random_state=random_state
)

with mlflow.start_run():

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # After training your model
    eval_data = X_test.copy()
    eval_data["target"] = y_test

    # Log the model with MLflow
    signature = infer_signature(X_train, model.predict(X_train))
    model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)  # type: ignore

    result = mlflow.models.evaluate(  # type: ignore
        model=model_info.model_uri,
        data=eval_data,
        targets="target",
        model_type="regressor",
    )

    # Then use SHAP directly with your preferred plot settings
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer(X_test)  # Returns Explanation object

    # Create plots without grouping
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, plot_type="bar", max_display=None, show=False)
    shap_bar_plot = plt.gcf()
    shap_bar_plot.tight_layout()
    mlflow.log_figure(shap_bar_plot, "shap_bar_plot.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, plot_type="dot", max_display=None, show=False)
    shap_dot_plot = plt.gcf()
    shap_dot_plot.tight_layout()
    mlflow.log_figure(shap_dot_plot, "shap_dot_plot.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, show=False)
    shap_beeswarm_plot = plt.gcf()
    shap_beeswarm_plot.tight_layout()
    mlflow.log_figure(shap_beeswarm_plot, "shap_beeswarm_plot.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.decision_plot(explainer.expected_value, shap_values.values, show=False)
    shap_decision_plot = plt.gcf()
    shap_decision_plot.tight_layout()
    mlflow.log_figure(shap_decision_plot, "shap_decision_plot.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    shap_waterfall_plot = plt.gcf()
    shap_waterfall_plot.tight_layout()
    mlflow.log_figure(shap_waterfall_plot, "shap_waterfall_plot.png")
    plt.close()

    # Log plots to MLflow if needed

    rmse = mean_squared_error(y_test, model.predict(X_test))
    mlflow.log_metric("rmse", rmse)

    # Calculate adjusted R² score
    r2 = r2_score(y_test, model.predict(X_test))
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    mlflow.log_metric("adjusted_r2", adjusted_r2)

    print(f"Mean Squared Error: {result.metrics['mean_squared_error']}")
    print(f"R^2 Score: {result.metrics['r2_score']}")
    print(f"Adjusted R^2 Score: {adjusted_r2}")
