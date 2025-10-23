import os
import mlflow
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from drl_experiment import VectorDRL, TrainEnvConfig, VectorDRLConfig
from utils import setup_logging


def setup_mlflow(all_params):
    mlflow_params = all_params["mlflow"]

    if not mlflow_params["use_mlflow"]:
        return
    
    match mlflow_params["record_platform"]:
        case "azure":
            path = os.path.join(os.path.dirname(__file__), "..", "config.json")
            print(path)
            ml_client = MLClient.from_config(
                credential=DefaultAzureCredential(),
                config_path=path
            )
            mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        case "dagshub":
            import dagshub
            dagshub.init(repo_owner='liverHawk', repo_name='research_data_drl', mlflow=True)
            mlflow_tracking_uri = mlflow_params["dagshub_url"]
        case "local":
            mlflow_tracking_uri = mlflow_params["local_url"]
        case _:
            raise ValueError(f"Invalid record platform: {mlflow_params['record_platform']}")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{mlflow_params['experiment_name']}_train_vector_drl"
    )
    mlflow.set_tags(all_params["sampling"]["method"])


def load_params():
    all_params = yaml.safe_load(open("params.yaml"))
    os.makedirs(os.path.abspath(os.path.join("result", "train_vector_drl")), exist_ok=True)
    setup_mlflow(all_params)
    return all_params


def main():
    params = load_params()
    logger = setup_logging(
        os.path.abspath(os.path.join("result", "log", "train_vector_drl.log"))
    )
    mlflow.pytorch.autolog()
    # mlflow.start_run()

    logger.info("Loading data...")
    train_env_config = TrainEnvConfig(
        reward_list=[0.5, 1.5, -1.0, -0.5, -1.0],
        max_steps=10_000,
        normalize_method="min-max",
        rolling_window=10,
    )
    vector_drl_config = VectorDRLConfig(
        device_number=int(params["cuda_device_number"]),
        train_data_path=os.path.join(os.path.dirname(__file__), "..", params["data_path"]["train"]),
        test_data_path=os.path.join(os.path.dirname(__file__), "..", params["data_path"]["test"]),
        train_env_config=train_env_config,
        use_mlflow=params["mlflow"]["use_mlflow"],
    )
    mlflow.log_params({
        "train_path": params["data_path"]["train"],
        "test_path": params["data_path"]["test"],
    })

    logger.info("Initializing VectorDRL...")
    vector_drl = VectorDRL(vector_drl_config)

    logger.info("Training...")
    vector_drl.train(n_steps=10000, loss_save_path=os.path.abspath(os.path.join("result", "train_vector_drl")))
    logger.info("Training completed.")

    logger.info("Saving model...")
    vector_drl.save_model(os.path.abspath(os.path.join("models")))
    logger.info("Model saved.")

    mlflow.end_run()
    logger.info("MLflow run completed.")


if __name__ == "__main__":
    main()
