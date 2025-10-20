import os
import sys
import mlflow
import yaml

import pandas as pd

from glob import glob
from utils import setup_logging
from classifier import ImprovedC45
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from lib.src.statistics import get_statistics

# mlflow.tracking.fluent.disable_logged_model()


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
        f"{mlflow_params['experiment_name']}_evaluate"
    )


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    os.makedirs("result/evaluate", exist_ok=True)

    return all_params, data_path


def load_data(data_path):
    files = glob(os.path.join(data_path, "*.csv.gz"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def evaluate(df, params, logger):
    model = ImprovedC45(
        load_path=os.path.abspath(os.path.join("model", "improved_c45_model.joblib"))
    )
    # samples = df.head()
    # signature = mlflow.models.infer_signature(
    #     samples.drop(columns=["Label"]), samples["Label"]
    # )

    # model_info = mlflow.sklearn.log_model(
    #     model.clf,
    #     name="improved_c45_model",
    #     signature=signature,
    #     registered_model_name="improved_c45_model"
    # )

    logger.info("Evaluating model...")
    X = df.drop(columns=["Label"])
    y = df["Label"]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    statistics, statistics_keys = get_statistics(len(y.unique()), y_pred_proba, y_pred, y)

    for key in statistics_keys:
        with open(f"result/evaluate/evaluate_{key}.txt", "w") as f:
            f.write(statistics[key])
        mlflow.log_artifact(f"result/evaluate/evaluate_{key}.txt", artifact_path="evaluate")
    
    logger.info("Evaluation completed.")


def main():
    params, data_path = load_params()
    logger = setup_logging(
        os.path.abspath(os.path.join("result", "log", "evaluate.log"))
    )
    df = load_data(data_path)
    evaluate(df, params, logger)


if __name__ == "__main__":
    main()