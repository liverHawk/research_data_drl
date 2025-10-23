import os
import mlflow
import yaml
import pandas as pd
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from drl_experiment import VectorDRL, TrainEnvConfig, VectorDRLConfig
from utils import setup_logging

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
        f"{mlflow_params['experiment_name']}_evaluate_vector_drl"
    )
    # all_params["sampling"]["method"] is a dictionary
    mlflow.set_tags(all_params["sampling"]["method"])



def load_params():
    os.makedirs("result/evaluate_vector_drl", exist_ok=True)
    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)
    return all_params


def _plot_confusion_matrix():
    df = pd.read_csv("result/evaluate_vector_drl/result.csv")
    cm = confusion_matrix(df["actual"], df["action"])
    
    same_shape_matrix = np.zeros(cm.shape)
    for i in range(cm.shape[1]):
        if cm[i, :].sum() == 0:
            continue
        same_shape_matrix[i, :] = cm[i, :] / float(cm[i, :].sum())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(same_shape_matrix, annot=True, fmt='.3f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    path = os.path.join("result", "evaluate_vector_drl", "confusion_matrix.png")
    plt.savefig(path)
    mlflow.log_artifact(path, artifact_path="evaluate_vector_drl")

def main():
    params = load_params()
    logger = setup_logging(
        os.path.abspath(os.path.join("result", "log", "evaluate_vector_drl.log"))
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
        device_number=params["cuda_device_number"],
        train_data_path=os.path.join(os.path.dirname(__file__), "..", params["data_path"]["train"]),
        test_data_path=os.path.join(os.path.dirname(__file__), "..", params["data_path"]["test"]),
        train_env_config=train_env_config,
    )

    logger.info("Initializing VectorDRL...")
    vector_drl = VectorDRL(vector_drl_config)
    logger.info("Loading model...")
    vector_drl.load_model(os.path.abspath(os.path.join("models")))
    logger.info("Model loaded.")

    logger.info("Evaluating...")
    result_list = vector_drl.test(split_size=params.get("split_size", 20))
    with open("result/evaluate_vector_drl/result.csv", "w") as f:
        f.write("action,actual\n")
        for result in result_list:
            f.write(f"{result[0]},{result[1]}\n")
    logger.info("Evaluation completed.")

    # 評価結果をメトリクスとして記録
    df_result = pd.read_csv("result/evaluate_vector_drl/result.csv")
    accuracy = (df_result['action'] == df_result['actual']).mean()
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_samples", len(df_result))
    
    # 各クラスの精度を計算
    for class_label in df_result['actual'].unique():
        class_mask = df_result['actual'] == class_label
        if class_mask.sum() > 0:
            class_accuracy = (df_result.loc[class_mask, 'action'] == df_result.loc[class_mask, 'actual']).mean()
            mlflow.log_metric(f"test_accuracy_class_{class_label}", class_accuracy)

    _plot_confusion_matrix()

    mlflow.end_run()
    logger.info("MLflow run completed.")


if __name__ == "__main__":
    main()
