import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# initialize DagsHub + set MLflow tracking URI
dagshub.init(
    repo_owner='Mervin50',
    repo_name='Swiggy_Delivery_Time_Prediction',  # ✅ corrected repo name
    mlflow=True
)

mlflow.set_tracking_uri("https://dagshub.com/Mervin50/Swiggy_Delivery_Time_Prediction.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info

if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # path to saved run info
    run_info_path = root_path / "run_information.json"

    # load run details
    run_info = load_model_information(run_info_path)
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]

    # full model URI
    model_registry_path = f"runs:/{run_id}/{model_name}"

    # register model in registry
    model_version = mlflow.register_model(
        model_uri=model_registry_path,
        name=model_name
    )

    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    # Promote model to Production stage
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Production",
        archive_existing_versions=True  # ✅ archives older production models
    )

    logger.info("Model successfully pushed to PRODUCTION stage")

