import os
import sys
import mlflow
import yaml

import pandas as pd

from glob import glob
from utils import setup_logging
from classifier import ImprovedC45


def make_dir():
    model_path = os.path.abspath("model")
    os.makedirs(model_path, exist_ok=True)


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    all_params = yaml.safe_load(open("params.yaml"))
    mlflow.set_tracking_uri(all_params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_train"
    )

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
    model.fit(X, y)
    logger.info("Model training completed.")

    logger.info("Logging model to MLflow...")
    model_path = os.path.abspath(os.path.join("model", "improved_c45_model.joblib"))
    model.save(model_path)


def main():
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


if __name__ == "__main__":
    main()