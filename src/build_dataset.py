import mlflow
import os
import yaml
import sys

import ipaddress as ip
import numpy as np
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import setup_logging, BASE, CICIDS2017
from csv_utils import save_split_csv, multiprocess_save_csv


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
        f"{all_params['mlflow']['experiment_name']}_build_dataset"
    )


def make_dir():
    path = os.path.abspath("data")
    os.makedirs(
        os.path.join(path, "train"), exist_ok=True
    )
    os.makedirs(
        os.path.join(path, "train", "raw"), exist_ok=True
    )
    os.makedirs(
        os.path.join(path, "test"), exist_ok=True
    )
    os.makedirs(
        os.path.join(path, "test", "raw"), exist_ok=True
    )
    os.makedirs(
        os.path.join(path, "..", "log"), exist_ok=True
    )


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python build_dataset.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    return all_params["build_dataset"], data_path


def fast_process(df, type="normal"):
    if type == "normal":
        df = df.drop(CICIDS2017().get_delete_columns(), axis=1)
        df = df.drop(columns=['Attempted Category'])
    elif type == "full":
        df = df.drop(['Flow ID', 'Src IP', 'Attempted Category'], axis=1)
        # Timestamp→秒
        df['Timestamp'] = (
            pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
            .astype('int64') // 10**9
        )
        # IP文字列→整数
        df['Dst IP'] = df['Dst IP'].apply(lambda x: int(ip.IPv4Address(x)))
    # 欠損／無限大落とし
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def load_data(path):
    files = glob(os.path.join(path, "*.csv"))
    dfs = [
        fast_process(pd.read_csv(f)) for f in files
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df


def column_adjustment(df):
    labels = df["Label"].unique()

    for label in labels:
        if "Attempted" in label:
            df.loc[df["Label"] == label, "Label"] = "BENIGN"
        if "Web Attack" in label:
            df.loc[df["Label"] == label, "Label"] = "Web Attack"
        if "Infiltration" in label:
            df.loc[df["Label"] == label, "Label"] = "Infiltration"
        if "DoS" in label and label != "DDoS":
            df.loc[df["Label"] == label, "Label"] = "DoS"
    
    rename_dict = {
        k: v for k, v in zip(
            CICIDS2017().get_features_labels(),
            BASE().get_features_labels()
        )
    }
    df = df.rename(columns=rename_dict)

    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])

    with open("label.txt", "w") as f:
        for k, v in enumerate(le.classes_):
            f.write(f"{k}\t{v}\n")
    
    return df


def save_csv(train_df, test_df, logger):
    train_path = os.path.abspath(os.path.join("data", "train", "raw"))
    test_path = os.path.abspath(os.path.join("data", "test", "raw"))

    train_array = save_split_csv(
        train_df,
        train_path,
        "train",
    )
    test_array = save_split_csv(
        test_df,
        test_path,
        "test",
    )
    combined_array_df = [df for df, _ in train_array + test_array]
    combined_array_path = [path for _, path in train_array + test_array]

    logger.info("Start saving CSV files")
    multiprocess_save_csv(combined_array_df, combined_array_path)



def main():
    make_dir()
    params, data_path = load_params()
    logger = setup_logging(
        os.path.join("log", "build_dataset.log")
    )
    mlflow.start_run()

    logger.info("Start building dataset")
    df = load_data(data_path)
    logger.info(f"Loaded data from {data_path}")
    logger.info(f"Start column adjustment")
    df = column_adjustment(df)
    logger.info(f"Column adjustment finished")

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["Label"]
    )
    logger.info(f"Train/Test split finished")

    save_csv(train_df, test_df, logger)
    
    mlflow.end_run()


if __name__ == "__main__":
    main()