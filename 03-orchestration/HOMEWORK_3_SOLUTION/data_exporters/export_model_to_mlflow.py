
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_model_to_mlflow(model, **kwargs):
    vectorizer, lr_model = model
    # Set MLflow tracking URI to the service running at http://mlflow:5050
    mlflow.set_tracking_uri("http://mlflow:5050")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the linear regression model
        mlflow.sklearn.log_model(lr_model, "linear_regression_model")

        # Save and log the DictVectorizer as an artifact
        import pickle
        with open("vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact("vectorizer.pkl", "vectorizer_artifact")

        # Optional: Print run ID for reference
        print(f"MLflow run ID: {run.info.run_id}")
