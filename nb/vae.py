#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation, rc
from glob import glob


# In[ ]:


path = "../data/train/binary"
files = glob(f"{path}/*.csv.gz")
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)
# df = df[df["Label"] == 0]
print(df.shape)
feature_cols = [c for c in df.columns if c != 'Label']
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

label_values_counts = len(df["Label"].unique())

train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Label"])


# In[ ]:


class DataFrameDataset(Dataset):
    def __init__(self, df, target_column):
        self.features = df.drop(columns=[target_column]).values.astype(np.float32)
        self.labels = df[target_column].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        if self.labels is not None:
            y = torch.tensor(self.labels[idx])
            return x, y
        else:
            return x

BATCH_SIZE = 500
train_dataset = DataFrameDataset(train_df, target_column='Label')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = DataFrameDataset(val_df, target_column='Label')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:


# BATCH_SIZE = 100

# trainval_data = MNIST("./test_data", 
#                    train=True, 
#                    download=True, 
#                    transform=transforms.ToTensor())

# train_size = int(len(trainval_data) * 0.8)
# val_size = int(len(trainval_data) * 0.2)
# train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

# train_loader = DataLoader(dataset=train_data,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=0)

# val_loader = DataLoader(dataset=val_data,
#                         batch_size=BATCH_SIZE,
#                         shuffle=True,
#                         num_workers=0)

# print("train data size: ",len(train_data))   #train data size:  48000
# print("train iteration number: ",len(train_data)//BATCH_SIZE)   #train iteration number:  480
# print("val data size: ",len(val_data))   #val data size:  12000
# print("val iteration number: ",len(val_data)//BATCH_SIZE)   #val iteration number:  120


# In[ ]:


images, labels = next(iter(train_loader))
print("images_size:",images.size())   #images_size: torch.Size([100, 1, 28, 28])
print("label:",labels[:10])   #label: tensor([7, 6, 0, 6, 4, 8, 5, 2, 2, 3])

# image_numpy = images.detach().numpy().copy()
# plt.imshow(image_numpy[0,0,:,:], cmap='gray')


# In[ ]:


class Encoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(88, 300)
    self.lr2 = nn.Linear(300, 100)
    self.lr_ave = nn.Linear(100, z_dim)   #average
    self.lr_dev = nn.Linear(100, z_dim)   #log(sigma^2)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.lr(x)
    x = self.relu(x)
    x = self.lr2(x)
    x = self.relu(x)
    ave = self.lr_ave(x)    #average
    log_dev = self.lr_dev(x)    #log(sigma^2)

    ep = torch.randn_like(ave)   #平均0分散1の正規分布に従い生成されるz_dim次元の乱数
    z = ave + torch.exp(log_dev / 2) * ep   #再パラメータ化トリック
    return z, ave, log_dev

class Decoder(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.lr = nn.Linear(z_dim, 100)
    self.lr2 = nn.Linear(100, 300)
    self.lr3 = nn.Linear(300, 88)
    self.relu = nn.ReLU()

  def forward(self, z):
    x = self.lr(z)
    x = self.relu(x)
    x = self.lr2(x)
    x = self.relu(x)
    x = self.lr3(x)
    x = torch.sigmoid(x)   #MNISTのピクセル値の分布はベルヌーイ分布に近いと考えられるので、シグモイド関数を適用します。
    return x

class VAE(nn.Module):
  def __init__(self, z_dim):
    super().__init__()
    self.encoder = Encoder(z_dim)
    self.decoder = Decoder(z_dim)

  def forward(self, x):
    z, ave, log_dev = self.encoder(x)
    x = self.decoder(z)
    return x, z, ave, log_dev


# In[ ]:


def criterion(predict, target, ave, log_dev):
  bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
  kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
  loss = bce_loss + kl_loss
  return loss


def extract_latent_vectors(model, dataloader, device=None, use_mean=True, n_samples=1):
    """
    dataloader から順に encoder を通して潜在ベクトルを取得して返す。
    - use_mean=True: encoder の平均 (ave) を返す（決定論的表現）
    - use_mean=False: ave, log_dev から n_samples 個サンプリングして平均を返す（確率的表現）
    戻り値: (Z, Y)  Z: (N, z_dim) numpy, Y: (N,) ラベル numpy
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for batch in dataloader:
            # DataFrameDataset の __getitem__ は (x, y) を返す想定
            x, y = batch
            x = x.to(device).view(x.size(0), -1).float()
            z_sample, ave, log_dev = model.encoder(x)  # encoder が (z, ave, log_dev) を返す前提
            if use_mean:
                z_out = ave
            else:
                samples = []
                for _ in range(n_samples):
                    eps = torch.randn_like(ave)
                    samples.append(ave + torch.exp(log_dev / 2) * eps)
                z_out = torch.stack(samples, dim=0).mean(dim=0)
            zs.append(z_out.cpu())
            ys.append(y.cpu())
    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()
    return Z, Y


def decode_latent_vectors(model, Z, device=None, batch_size=512):
    """
    Z: numpy array (N, z_dim) または torch.Tensor (N, z_dim)
    戻り値: reconstructions numpy array shape (N, n_features) (float32, 値域はモデル出力のスケール)
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    if isinstance(Z, np.ndarray):
        Z = torch.from_numpy(Z).to(torch.float32)
    Z = Z.to(device)
    recons = []
    with torch.no_grad():
        for i in range(0, Z.size(0), batch_size):
            z_batch = Z[i:i + batch_size].to(device).float()
            x_hat = model.decoder(z_batch)  # decoder は (batch, n_features) を返す想定
            recons.append(x_hat.cpu())
    recons = torch.cat(recons, dim=0).numpy()
    return recons

def inverse_scale_reconstructions(recons, scaler):
    """
    recons: numpy array (N, n_features) -- decoder 出力（通常は 0..1）
    scaler: 学習時に用いた sklearn の scaler (例: MinMaxScaler)
    戻り値: 元スケールに戻した numpy array (N, n_features)
    """
    # scaler は fit されている前提
    return scaler.inverse_transform(recons)


def reconstruct_from_latents(Z, model, scaler, feature_cols, device=None, batch_size=512):
    """
    Z: numpy (N, z_dim) または torch.Tensor (N, z_dim)
    model: VAE モデル（decoder を持つ）
    scaler: 学習時に使った sklearn スケーラ（MinMaxScaler 等）
    feature_cols: 元データの特徴量カラム名リスト（Label を除く順序）
    戻り値: pandas.DataFrame (N, len(feature_cols)) 元スケールに戻した再構成特徴量
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    # デコード（0..1 スケール想定）
    recons_scaled = decode_latent_vectors(model, Z, device=device, batch_size=batch_size)
    # スケール逆変換して元の値に戻す
    recons_original = inverse_scale_reconstructions(recons_scaled, scaler)
    # DataFrame に変換（feature_cols の順序に合わせる）
    df_recon = pd.DataFrame(recons_original, columns=feature_cols)
    return df_recon

def reconstruct_single(z, model, scaler, feature_cols, device=None):
    """
    単一潜在ベクトル z (1D numpy or torch) を再構成して DataFrame で返す
    """
    if isinstance(z, np.ndarray):
        Z = z.reshape(1, -1)
    elif isinstance(z, torch.Tensor):
        Z = z.detach().cpu().numpy().reshape(1, -1)
    else:
        Z = np.array(z).reshape(1, -1)
    return reconstruct_from_latents(Z, model, scaler, feature_cols, device=device, batch_size=1)


# In[ ]:


z_dim = 2
num_epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = VAE(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}

for epoch in range(num_epochs):
  model.train()
  for i, (x, labels) in enumerate(train_loader):
    input = x.to(device).view(-1, 88).to(torch.float32)
    output, z, ave, log_dev = model(input)

    history["ave"].append(ave)
    history["log_dev"].append(log_dev)
    history["z"].append(z)
    history["labels"].append(labels)
    loss = criterion(output, input, ave, log_dev)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
      print(f'\rEpoch: {epoch+1:3d}, loss: {loss: 04.4f}', end='')
    history["train_loss"].append(loss)

  model.eval()
  with torch.no_grad():
    for i, (x, labels) in enumerate(val_loader):
      input = x.to(device).view(-1, 88).to(torch.float32)
      output, z, ave, log_dev = model(input)

      loss = criterion(output, input, ave, log_dev)
      history["val_loss"].append(loss)

    print(f' -> val_loss: {loss: 0.4f}')

  scheduler.step()


# In[ ]:


train_loss_tensor = torch.stack(history["train_loss"])
train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(train_loss_np)


# In[ ]:


val_loss_tensor = torch.stack(history["val_loss"])
val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(val_loss_np)


# In[ ]:


ave_tensor = torch.cat(history["ave"])
log_var_tensor = torch.cat(history["log_dev"])
z_tensor = torch.cat(history["z"])
labels_tensor = torch.cat(history["labels"])
print(ave_tensor.size())   #torch.Size([9600, 100, 2])
print(log_var_tensor.size())   #torch.Size([9600, 100, 2])
print(z_tensor.size())   #torch.Size([9600, 100, 2])
print(labels_tensor.size())   #torch.Size([9600, 100])

ave_np = ave_tensor.to('cpu').detach().numpy().copy()
log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
z_np = z_tensor.to('cpu').detach().numpy().copy()
labels_np = labels_tensor.to('cpu').detach().numpy().copy()
print(ave_np.shape)   #(9600, 100, 2)
print(log_var_np.shape)   #(9600, 100, 2)
print(z_np.shape)   #(9600, 100, 2)
print(labels_np.shape)   #(9600, 100)


# In[ ]:


label_values_counts


# In[ ]:


map_keyword = "tab10"
cmap = plt.get_cmap(map_keyword)

plt.figure(figsize=[10,10])
for label in range(label_values_counts):
    x = z_np[labels_np == label, 0]
    y = z_np[labels_np == label, 1]
    plt.scatter(x, y, color=cmap(label/label_values_counts), label=label, s=15)
    if len(x) > 0 and len(y) > 0:
        plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
    plt.legend(loc="upper left")
    plt.show()


# In[ ]:


batch_num = 9580
plt.figure(figsize=[10,10])
for label in range(label_values_counts):
  x = z_np[labels_np == label, 0]
  y = z_np[labels_np == label, 1]
  plt.scatter(x, y, color=cmap(label/label_values_counts), label=label, s=15)
  plt.annotate(label, xy=(np.mean(x),np.mean(y)),size=20,color="black")
plt.legend(loc="upper left")


# In[ ]:


model.to("cpu")


Z_train, Y_train = extract_latent_vectors(
  model, train_loader, device=device, use_mean=True
)
df_recon = reconstruct_from_latents(
  Z_train, model, scaler, feature_cols, device=device
)
df_recon['Label'] = Y_train  # ラベルを付ける場合
df_recon.to_csv("reconstructed_train.csv", index=False)

model_device = next(model.parameters()).device  # モデルが置かれているデバイスを取得

Z_train, Y_train = extract_latent_vectors(
    model, train_loader, device=model_device, use_mean=True
)
df_recon = reconstruct_from_latents(
    Z_train, model, scaler, feature_cols, device=model_device
)
df_recon['Label'] = Y_train
df_recon.to_csv("reconstructed_train.csv", index=False)


