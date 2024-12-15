import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set experiment name
mlflow.set_experiment("My First Experiment")

# Start an MLflow run
with mlflow.start_run():
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Run completed. Check your MLflow UI for details.")
