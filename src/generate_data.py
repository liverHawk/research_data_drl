import torch
import mlflow
import os
import cProfile
import pstats

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from vae_gan import VAEGAN  # VAE-GANの実装をインポート
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml


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
        f"{all_params['mlflow']['experiment_name']}_generate_data"
    )


def load_data():
    path = "data/train/binary"
    files = glob(f"{path}/*.csv.gz")
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)

    params = yaml.safe_load(open("params.yaml"))
    return data, params


def generate_data(df):
    normal_x = df[df["Label"] == 0].drop(columns=["Label"])

    input_dim = len(normal_x.columns)
    latent_dim = 20
    batch_size = 64

    # 正常データ: normal_x (Tensor, shape=[batch, input_dim])
    vae_gan_origin = VAEGAN(input_dim, latent_dim)
    vae_gan_origin.eval()
    with torch.no_grad():
        mu, logvar = vae_gan_origin.encoder(normal_x)
        z = vae_gan_origin.reparameterize(mu, logvar)
        recon_x = vae_gan_origin.decoder(z)
    torch.save(vae_gan_origin.state_dict(), "vae_gan_model.pth")
    
    labels = df["Label"].unique()
    labels.remove(0)

    for label in labels:
        vae_gan = VAEGAN(input_dim, latent_dim)
        vae_gan.load_state_dict(torch.load("vae_gan_model.pth"))
        attack_x = df[df["Label"] == label].drop(columns=["Label"])
        
        # 攻撃データ: attack_x (Tensor, shape=[batch, input_dim])
        optimizer_D = optim.Adam(vae_gan.discriminator.parameters())
        optimizer_G = optim.Adam(list(vae_gan.encoder.parameters()) + list(vae_gan.decoder.parameters()))

        # 判別器の学習
        real_score = vae_gan.discriminator(attack_x)
        fake_score = vae_gan.discriminator(recon_x.detach())
        bce = nn.BCELoss()
        real_labels = torch.ones_like(real_score)
        fake_labels = torch.zeros_like(fake_score)
        d_loss = bce(real_score, real_labels) + bce(fake_score, fake_labels)
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 生成器（VAE部分）の学習
        fake_score = vae_gan.discriminator(recon_x)
        g_loss = bce(fake_score, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()



        vae_gan.eval()
        with torch.no_grad():
            z_attack = torch.randn(batch_size, latent_dim)  # 攻撃用潜在ベクトル
            generated_attack = vae_gan.decoder(z_attack)

        mlflow.log_artifact(generated_attack, artifact_path="generate_data")
        mlflow.log_artifact(z_attack, artifact_path="generate_data")
        mlflow.log_artifact(recon_x, artifact_path="generate_data")
        mlflow.log_artifact(attack_x, artifact_path="generate_data")
        mlflow.log_artifact(real_score, artifact_path="generate_data")
        mlflow.log_artifact(fake_score, artifact_path="generate_data")
        mlflow.log_artifact(real_labels, artifact_path="generate_data")
        mlflow.log_artifact(fake_labels, artifact_path="generate_data")


def main():
    # with cProfile.Profile() as pr:
    data, params = load_data()
    setup_mlflow(params)

    mlflow.start_run()
    generate_data(data)
    mlflow.end_run()
    
    # with open("generate_data.prof", "w") as f:
    #     ps = pstats.Stats(pr, stream=f)
    #     ps.sort_stats("cumulative")
    #     ps.print_stats()
    # with open("generate_data.prof", "w") as f:
    #     ps = pstats.Stats(pr, stream=f)
    #     ps.sort_stats("time")
    #     ps.print_stats()



if __name__ == "__main__":
    main()