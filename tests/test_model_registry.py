import pytest
import mlflow
from mlflow import MlflowClient
import dagshub
import json

# Initialize DagsHub and MLflow tracking URI
dagshub.init(repo_owner='Mervin50',
             repo_name='Swiggy_Delivery_Time_Prediction',
             mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Mervin50/Swiggy_Delivery_Time_Prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


def ensure_model_in_staging(model_name):
    client = MlflowClient()
    # Get all versions of the model
    all_versions = client.get_latest_versions(name=model_name)

    if not all_versions:
        raise ValueError(f"No versions found for model '{model_name}'")

    # Check if any version is already in Staging
    staging_versions = [v for v in all_versions if v.current_stage == "Staging"]

    if staging_versions:
        # Model already in staging, return the first staging version number
        return staging_versions[0].version

    # Otherwise, promote the latest version to Staging
    # Assuming the highest version number is the latest
    latest_version = max(all_versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Staging",
        archive_existing_versions=True
    )
    print(f"Promoted model version {latest_version.version} to Staging")
    return latest_version.version


# Load model name from JSON
model_name = load_model_information("run_information.json")["model_name"]


@pytest.mark.parametrize(argnames="model_name, stage",
                         argvalues=[(model_name, "Staging")])
def test_load_model_from_registry(model_name, stage):
    # Ensure model is in staging before testing
    version = ensure_model_in_staging(model_name)

    client = MlflowClient()
    latest_versions = client.get_latest_versions(name=model_name, stages=[stage])
    latest_version = latest_versions[0].version if latest_versions else None

    assert latest_version is not None, f"No model at {stage} stage"

    # Load model from registry path
    model_path = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_path)

    assert model is not None, "Failed to load model from registry"
    print(f"The {model_name} model with version {latest_version} was loaded successfully")
