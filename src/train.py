import os
import sys
import mlflow
import yaml
import cProfile
import pstats

import pandas as pd

from glob import glob
from utils import setup_logging
from classifier import ImprovedC45
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def setup_mlflow(all_params):
    if all_params["mlflow"]["use_azure"]:
        path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        print(path)
        ml_client = MLClient.from_config(
            credential=DefaultAzureCredential(),
            config_path=path
        )
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    else:
        mlflow_tracking_uri = all_params["mlflow"]["tracking_uri"]
    if all_params["mlflow"]["use_dagshub"]:
        import dagshub
        dagshub.init(repo_owner='liverHawk', repo_name='research_data_drl', mlflow=True)
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_train"
    )


def make_dir():
    model_path = os.path.abspath("model")
    os.makedirs(model_path, exist_ok=True)


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    return all_params["train"], data_path


def load_data(data_path):
    files = glob(os.path.join(data_path, "*.csv.gz"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def train(df, params, logger):
    mlflow.sklearn.autolog()
    model = ImprovedC45(
        max_depth=params.get("max_depth", 3),
    )
    X = df.drop(columns=["Label"])
    y = df["Label"]

    logger.info("Training model...")
    print(X.shape)
    model.fit(X, y)
    logger.info("Model training completed.")

    logger.info("Logging model to MLflow...")
    model_path = os.path.abspath(os.path.join("model", "improved_c45_model.joblib"))
    model.save(model_path)


def main():
    # with cProfile.Profile() as pr:
    make_dir()
    params, data_path = load_params()
    logger = setup_logging(
        os.path.abspath(os.path.join("log", "train.log"))
    )

    mlflow.start_run()

    logger.info("Loading data...")
    df = load_data(data_path)
    logger.info(f"Data shape: {df.shape}")

    train(df, params, logger)

    mlflow.end_run()
    
    # with open("train.prof", "w") as f:
    #     ps = pstats.Stats(pr, stream=f)
    #     ps.sort_stats("cumulative")
    #     ps.print_stats()
    # with open("train.prof", "w") as f:
    #     ps = pstats.Stats(pr, stream=f)
    #     ps.sort_stats("time")
    #     ps.print_stats()


if __name__ == "__main__":
    main()