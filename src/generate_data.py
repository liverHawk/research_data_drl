import torch

import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from vae_gan import VAEGAN  # VAE-GANの実装をインポート



def load_data():
    path = "data/train/binary"
    files = glob(f"{path}/*.csv.gz")
    dfs = [pd.read_csv(f) for f in files]
    data = pd.concat(dfs, ignore_index=True)
    return data


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


def main():
    data = load_data()
    generate_data(data)



if __name__ == "__main__":
    main()