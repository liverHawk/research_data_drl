import os
import mlflow
import yaml

import numpy as np
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from glob import glob

from utils import setup_logging
import over_sampling
import under_sampling
from csv_utils import save_split_csv, multiprocess_save_csv


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
        f"{mlflow_params['experiment_name']}_sampling"
    )


def make_dir():
    train_binary_path = os.path.abspath(os.path.join("data", "train", "binary"))
    os.makedirs(train_binary_path, exist_ok=True)
    test_binary_path = os.path.abspath(os.path.join("data", "test", "binary"))
    os.makedirs(test_binary_path, exist_ok=True)


def load_params():
    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    if not all_params["sampling"]["use_sampling"]:
        # code finish
        print("Sampling finished")
        exit(0)
    
    return all_params["sampling"]["method"]


def load_data(phase="train", data_type="raw"):
    files = glob(os.path.join(f"data/{phase}/{data_type}", "*.csv.gz"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def split_data(df: pd.DataFrame, limit_under_place: int, _type: str):
    if limit_under_place == 1:
        return df, None

    value_counts = df["Label"].value_counts()
    if _type == "under":
        except_label = value_counts.index[-limit_under_place:]
        except_df = df[df["Label"].isin(except_label)]
        sampling_df = df[~df["Label"].isin(except_label)]
        return sampling_df, except_df

    elif _type == "over":
        except_label = value_counts.index[:limit_under_place]
        except_df = df[df["Label"].isin(except_label)]
        sampling_df = df[~df["Label"].isin(except_label)]
        return sampling_df, except_df

    else:
        raise ValueError(f"Invalid type: {_type}")


def call_over_sampling(df: pd.DataFrame, method_dict: dict):
    limit_over_place = method_dict["limit_place"]
    df, except_df = split_data(df, limit_over_place, "over")

    sampling_props = over_sampling.SamplingProps(df, "Label")
    match method_dict["name"]:
        case "random":
            resample_df, props = over_sampling.random(sampling_props)
        case "smote":
            resample_df, props = over_sampling.smote(sampling_props)
        case "adasyn":
            resample_df, props = over_sampling.adasyn(sampling_props)
        case "borderline_smote":
            resample_df, props = over_sampling.borderline_smote(sampling_props)
        case "kmeans_smote":
            resample_df, props = over_sampling.kmeans_smote(sampling_props)
        case "svmsmote":
            resample_df, props = over_sampling.svmsmote(sampling_props)
        case _:
            raise ValueError(f"Invalid sampling method: {method_dict['name']}")
    
    resample_df = resample_df.replace([np.inf, -np.inf], np.nan).dropna()
    return_df = pd.concat([resample_df, except_df])
    return return_df, props


def call_under_sampling(df: pd.DataFrame, method_dict: dict):
    limit_under_place = method_dict["limit_place"]

    df, except_df = split_data(df, limit_under_place, "under")

    sampling_props = under_sampling.SamplingProps(df, "Label")
    match method_dict["name"]:
        case "random":
            resample_df, props = under_sampling.random(sampling_props)
        case "cluster_centroid":
            resample_df, props = under_sampling.cluster_centroid(sampling_props)
        case "edited_nearest_neighbors":
            resample_df, props = under_sampling.edited_nearest_neighbors(sampling_props)
        case "repeated_edited_nearest_neighbors":
            resample_df, props = under_sampling.repeated_edited_nearest_neighbors(sampling_props)
        case "all_knn":
            resample_df, props = under_sampling.all_knn(sampling_props)
        case "instance_hardness_threshold":
            resample_df, props = under_sampling.instance_hardness_threshold(sampling_props)
        case "near_miss":
            resample_df, props = under_sampling.near_miss(sampling_props)
        case "neighbourhood_cleaning_rule":
            resample_df, props = under_sampling.neighbourhood_cleaning_rule(sampling_props)
        case "one_sided_selection":
            resample_df, props = under_sampling.one_sided_selection(sampling_props)
        case "tomek_links":
            resample_df, props = under_sampling.tomek_links(sampling_props)
        case _:
            raise ValueError(f"Invalid sampling method: {method_dict['name']}")
    
    return_df = pd.concat([resample_df, except_df])
    return return_df, props


def sampling(params, logger):
    logger.info("Loading data")
    df_binary = load_data("train", "binary")
    df_raw = load_data("train", "raw")
    logger.info("Data loaded")

    if params["type"] == "over":
        logger.info("Start sampling")
        resample_binary, props = call_over_sampling(df_binary, params)
        mlflow.log_param("binary_props", props)
        resample_raw, props = call_over_sampling(df_raw, params)
        mlflow.log_param("raw_props", props)
    elif params["type"] == "under":
        logger.info("Start sampling")
        resample_binary, props = call_under_sampling(df_binary, params)
        mlflow.log_param("binary_props", props)
        resample_raw, props = call_under_sampling(df_raw, params)
        mlflow.log_param("raw_props", props)
    else:
        raise ValueError(f"Invalid sampling method: {params['type']}")
    
    mlflow.log_params(params)
    logger.info("Sampling finished")

    logger.info("Start saving data")
    binary_path = os.path.abspath(os.path.join("data", "train", "sampled_binary"))
    raw_path = os.path.abspath(os.path.join("data", "train", "sampled_raw"))
    os.makedirs(binary_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    
    binary_array = save_split_csv(resample_binary, binary_path, "sampled_binary")
    raw_array = save_split_csv(resample_raw, raw_path, "sampled_raw")
    combined_array_df = [df for df, _ in binary_array + raw_array]
    combined_array_path = [path for _, path in binary_array + raw_array]

    logger.info("Saving data")
    multiprocess_save_csv(combined_array_df, combined_array_path)
    logger.info("Data saved")


def main():
    params = load_params()
    logger = setup_logging(
        os.path.join("result", "log", "sampling.log")
    )

    mlflow.start_run()
    sampling(params, logger)
    mlflow.end_run()


if __name__ == "__main__":
    main()