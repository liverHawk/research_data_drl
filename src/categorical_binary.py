import os
import sys
import mlflow
import yaml

from glob import glob
import pandas as pd
from utils import setup_logging
from csv_utils import save_split_csv, multiprocess_save_csv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


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
        f"{all_params['mlflow']['experiment_name']}_categorical_binary"
    )


def make_dir():
    train_binary_path = os.path.abspath(os.path.join("data", "train", "binary"))
    os.makedirs(train_binary_path, exist_ok=True)
    test_binary_path = os.path.abspath(os.path.join("data", "test", "binary"))
    os.makedirs(test_binary_path, exist_ok=True)


def load_params():
    all_params = yaml.safe_load(open("params.yaml"))
    setup_mlflow(all_params)

    return all_params["categorical_binary"]


def load_data(_type="train"):
    files = glob(os.path.join(f"data/{_type}/raw", "*.csv.gz"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def change_columns(df, logger):
    """
    Seriesを一気にフォーマットして文字列にし、各文字を個別の列として保存
    """
    new_df = df.copy()
    
    # Labelカラムは除外
    feature_columns = [
        {"type": "Destination Port", "max": 65535},
        {"type": "Protocol", "max": 255}
    ]

    for col in feature_columns:
        col_name = col["type"]
        
        # 数値型でない場合はスキップ
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            continue
            
        # 整数値に変換（小数点以下は切り捨て）
        int_values = df[col_name].astype(int)

        # 最大値から必要な桁数を決定
        max_val = col["max"]
        max_val_binary = format(max_val, 'b')
        num_digits = len(max_val_binary)
        
        # Series全体を一気に2進数文字列にフォーマット（ゼロパディング）  
        binary_strings = int_values.apply(lambda x: format(x, f'0{num_digits}b'))
        
        # 各文字位置を個別の列として作成
        for char_pos in range(num_digits):
            bit_col_name = f"{col_name}_{char_pos}"
            # 各文字列の指定位置の文字を取得して整数に変換
            new_df[bit_col_name] = binary_strings.str[char_pos].astype(int)
        
        # 元の列を削除
        new_df = new_df.drop(columns=[col_name])

        logger.info(f"列 '{col_name}' を {num_digits} 文字に分割しました")

    return new_df


def save_csv(df, logger, _type="train"):
    path = os.path.abspath(
        os.path.join("data", _type, "binary")
    )
    binary_array = save_split_csv(
        df,
        path,
        f"{_type}_binary",
    )
    binary_array_df = [df for df, _ in binary_array]
    binary_array_path = [path for _, path in binary_array]

    logger.info("Start saving CSV files")
    multiprocess_save_csv(binary_array_df, binary_array_path)


def main():
    make_dir()
    params = load_params()
    logger = setup_logging(
        os.path.join("log", "categorical_binary.log")
    )
    mlflow.start_run()

    logger.info(f"Loading data from data/train/raw")
    df = load_data()
    logger.info(f"Data loaded with shape: {df.shape}")

    logger.info("Start categorical to binary conversion")
    df = change_columns(df, logger)
    logger.info(f"Categorical to binary conversion finished with shape: {df.shape}")
    save_csv(df, logger)

    logger.info("Loading data from data/test/raw")
    df = load_data(_type="test")
    logger.info(f"Data loaded with shape: {df.shape}")

    logger.info("Start categorical to binary conversion")
    df = change_columns(df, logger)
    logger.info(f"Categorical to binary conversion finished with shape: {df.shape}")
    save_csv(df, logger, _type="test")
    
    mlflow.end_run()



if __name__ == "__main__":
    main()