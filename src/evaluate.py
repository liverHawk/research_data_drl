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

# mlflow.tracking.fluent.disable_logged_model()


def setup_mlflow(all_params):
    if all_params["mlflow"]["use_azure"]:
        import dagshub
        dagshub.init(repo_owner='liverHawk', repo_name='research_data_drl', mlflow=True)
        path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        print(path)
        ml_client = MLClient.from_config(
            credential=DefaultAzureCredential(),
            config_path=path
        )
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    else:
        mlflow_tracking_uri = all_params["mlflow"]["tracking_uri"]
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(
        f"{all_params['mlflow']['experiment_name']}_evaluate"
    )


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    return all_params["evaluate"], data_path


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
    samples = df.head()
    signature = mlflow.models.infer_signature(
        samples.drop(columns=["Label"]), samples["Label"]
    )

    model_info = mlflow.sklearn.log_model(
        model.clf,
        name="improved_c45_model",
        signature=signature,
        registered_model_name="improved_c45_model"
    )

    logger.info("Evaluating model...")
    result = mlflow.evaluate(
        model_info.model_uri,
        df,
        targets="Label",
        model_type="classifier",
        evaluators=["default"],
    )
    logger.info("Evaluation completed.")

    logger.info(f"Evaluation results: {result.metrics}")

    # path = os.path.abspath(os.path.join("log", "evaluation_report.csv"))
    # with open(path, "w") as f:
    #     f.write(result.metrics)


def main():
    params, data_path = load_params()
    logger = setup_logging(
        os.path.abspath(os.path.join("log", "evaluate.log"))
    )
    df = load_data(data_path)
    evaluate(df, params, logger)


if __name__ == "__main__":
    main()