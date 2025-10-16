import os
import sys
import torch
import yaml
import pandas as pd
import numpy as np
import cProfile
import pstats

from itertools import count
from glob import glob
from tqdm import tqdm

import flow_package as fp
from flow_package.multi_df_env import MultiDfEnv, EnvConfig

from utils import setup_logging, rolling_normalize
from network import DeepFlowNetwork
from network_v2 import DeepFlowNetworkV2

# 追加インポート: mlflow とプロットユーティリティ
import mlflow
import plot as plot_lib
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


# 追加: デバイス設定を学習コードと同じロジックで統一
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_params():
    if len(sys.argv) != 2:
        print("Usage: python src/evaluate_drl.py <input_file_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    all_params = yaml.safe_load(open("params.yaml"))

    setup_mlflow(all_params)
    # params = all_params["evaluate_drl"]
    return all_params, input_path


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
        f"{all_params['mlflow']['experiment_name']}_evaluate_drl"
    )


def load_csv(input):
    files = glob(os.path.join(input, "*.csv.gz"))
    df = pd.concat([
        pd.read_csv(f) for f in files
    ])
    df = df.reset_index(drop=True)
    df = rolling_normalize(df)
    return df


def write_result(cm_memory, prefix):
    # cm_memory: list of [predicted, actual]
    cm_data = []
    for pred, actual in cm_memory:
        pred_val = pred.item() if hasattr(pred, "item") else int(pred)
        actual_val = actual.item() if hasattr(actual, "item") else int(actual)
        cm_data.append([pred_val, actual_val])

    if len(cm_data) == 0:
        # 空の場合は空のファイルを作るだけ
        os.makedirs("evaluate", exist_ok=True)
        cm_path = os.path.join("evaluate", f"{prefix}_confusion_matrix.csv")
        pd.DataFrame(columns=["Predicted", "Actual"]).to_csv(cm_path, index=False)
        return None, 0.0, cm_path

    # クラス数はデータ上の最大ラベルから推定
    preds = [p for p, a in cm_data]
    actuals = [a for p, a in cm_data]
    n = max(max(preds), max(actuals)) + 1
    cm = np.zeros((n, n), dtype=int)

    # plot.py のラベル付け（x: Actual, y: Predicted）に合わせて cm[predicted][actual] を増やす
    for pred, actual in cm_data:
        cm[pred][actual] += 1

    os.makedirs("evaluate", exist_ok=True)
    cm_path = os.path.join("evaluate", f"{prefix}_confusion_matrix.csv")
    # CSV 保存（行: Predicted, 列: Actual のマトリクス形式）
    pd.DataFrame(cm).to_csv(cm_path, index=True)

    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0
    return cm, accuracy, cm_path


def to_tensor(state, include_category=True):
    # 学習側と同じように device を渡す
    if include_category:
        return fp.to_tensor(state, device=device)
    else:
        return torch.tensor(state, device=device, dtype=torch.float32)


def test(df, all_params):
    drl_options = all_params.get("drl_options", {})
    params = all_params.get("evaluate_drl", {})
    
    input = EnvConfig(
        data=df,
        label_column="Label",
        render_mode=None,
        window_size=drl_options.get("window_size", 10),
        max_steps=drl_options.get("max_steps", 100),
        normalize_method="rolling",
        rolling_window=drl_options.get("rolling_window", 10),
        test_mode=True,
    )
    env = MultiDfEnv(input)

    print(params)
    include_category = params.get("include_category", True)

    # 環境から正しい次元を取得してモデルを同じ呼び出し方で作る
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if include_category:
        network = DeepFlowNetwork(n_states, n_actions).to(device)
    else:
        network = DeepFlowNetworkV2(n_states, n_actions).to(device)

    # 学習側で保存したファイル名に合わせて読み込む
    if include_category:
        path = os.path.join("model", "drl_model_with_category.pth")
    else:
        path = os.path.join("model", "drl_model.pth")

    # デバイスを考慮してロード
    network.load_state_dict(torch.load(path, map_location=device))
    network.eval()

    cm_memory = []

    log_path = os.path.join("log", "evaluate_drl.log")
    logger = setup_logging(log_path)
    logger.info("Starting evaluation...")

    sum_rewards = 0.0
    
    for i_loop in range(1):
        raw_state, _ = env.reset()
        try:
            state = to_tensor(raw_state, include_category=include_category)
        except Exception as e:
            raise ValueError(f"Error converting state to tensor: {e}")
        progress_bar = tqdm(range(len(df)), desc=f"Evaluation Loop {i_loop+1}")
        for t in count():
            with torch.no_grad():
                predicted_action = network(state)
                if predicted_action.dim() == 1:
                    predicted_action = predicted_action.unsqueeze(0)
                predicted_action = predicted_action.max(1)[1].view(1, 1)

            raw_next_state, reward, terminated, _, info = env.step(predicted_action.item())
            sum_rewards += reward
            # predicted と actual を混同行列用に保存（predicted, actual）
            actual_answer = int(info["confusion_matrix_index"][1])
            cm_memory.append([predicted_action, actual_answer])

            if terminated:
                break
            try:
                next_state = to_tensor(raw_next_state, include_category=include_category)
            except Exception as e:
                raise ValueError(f"Error converting next state to tensor: {e}")

            state = next_state
            progress_bar.update(1)
        if i_loop > 0 and (i_loop + 1) % 5_000 == 0:
            mlflow.log_trace("evaluate_drl", f"Step {i_loop+1}", {"sum_rewards": sum_rewards})
        progress_bar.close()

    logger.info("Evaluation completed.")

    # 結果保存と mlflow ログ
    cm, accuracy, cm_csv_path = write_result(cm_memory, "evaluate")
    # 数値メトリクスを mlflow に保存（数値のみ）
    total_samples = int(cm.sum()) if cm is not None else 0
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("total_samples", int(total_samples))

    # 混同行列をプロットして保存・アップロード
    os.makedirs("evaluate/plots", exist_ok=True)
    if cm is not None:
        cm_img_path = os.path.join("evaluate/plots", "confusion_matrix_evaluate.png")
        plot_lib.plot_data(cm, "confusion_matrix", save_path=cm_img_path, fmt=".0f")
        mlflow.log_artifact(cm_img_path, artifact_path="evaluate/plots")

    # CSV もアーティファクトとして保存
    mlflow.log_artifact(cm_csv_path, artifact_path="evaluate")

    logger.info(f"Saved evaluation artifacts. accuracy={accuracy:.6f}, samples={total_samples}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <test_data_directory>")
        sys.exit(1)
    params, input_path = load_params()
    with cProfile.Profile() as pr:
        mlflow.start_run()

        df = load_csv(input_path)

        test(df, params)
        mlflow.end_run()
    
    with open("evaluate_drl.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats()
    with open("evaluate_drl.prof", "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("time")
        ps.print_stats()


if __name__ == "__main__":
    main()