import os
import sys
import mlflow
import yaml
import cProfile
import pstats

import pandas as pd

from glob import glob
from utils import setup_logging
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report
from classifier import ImprovedC45
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# mlflow.tracking.fluent.disable_logged_model()


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
    result = model.predict_proba(X)

    cm = multilabel_confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("total_samples", len(y))
    mlflow.log_dict(cm, artifact_file="evaluate_cm.json")
    mlflow.log_dict(report, artifact_file="evaluate_report.json")
    
    logger.info("Evaluation completed.")


def main():
    with cProfile.Profile() as pr:
        params, data_path = load_params()
        logger = setup_logging(
            os.path.abspath(os.path.join("log", "evaluate.log"))
        )
        df = load_data(data_path)
        evaluate(df, params, logger)
    
    with open("evaluate.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
    with open("evaluate.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("time")
        ps.print_stats()


if __name__ == "__main__":
    main()