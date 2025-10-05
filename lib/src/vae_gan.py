import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder: 入力ベクトル→潜在空間
class Encoder(nn.Module):
	def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64]):
		super().__init__()
		layers = []
		last_dim = input_dim
		for h in hidden_dims:
			layers.append(nn.Linear(last_dim, h))
			layers.append(nn.ReLU())
			last_dim = h
		self.net = nn.Sequential(*layers)
		self.fc_mu = nn.Linear(last_dim, latent_dim)
		self.fc_logvar = nn.Linear(last_dim, latent_dim)

	def forward(self, x):
		h = self.net(x)
		mu = self.fc_mu(h)
		logvar = self.fc_logvar(h)
		return mu, logvar

# Decoder: 潜在空間→元のベクトル
class Decoder(nn.Module):
	def __init__(self, latent_dim, output_dim, hidden_dims=[64, 128]):
		super().__init__()
		layers = []
		last_dim = latent_dim
		for h in hidden_dims:
			layers.append(nn.Linear(last_dim, h))
			layers.append(nn.ReLU())
			last_dim = h
		layers.append(nn.Linear(last_dim, output_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, z):
		return self.net(z)

# Discriminator: 入力ベクトル→本物/偽物
class Discriminator(nn.Module):
	def __init__(self, input_dim, hidden_dims=[128, 64]):
		super().__init__()
		layers = []
		last_dim = input_dim
		for h in hidden_dims:
			layers.append(nn.Linear(last_dim, h))
			layers.append(nn.ReLU())
			last_dim = h
		layers.append(nn.Linear(last_dim, 1))
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return torch.sigmoid(self.net(x))

# VAE-GAN全体をまとめるクラス例
class VAEGAN(nn.Module):
	def __init__(self, input_dim, latent_dim):
		super().__init__()
		self.encoder = Encoder(input_dim, latent_dim)
		self.decoder = Decoder(latent_dim, input_dim)
		self.discriminator = Discriminator(input_dim)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def forward(self, x):
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		recon_x = self.decoder(z)
		return recon_x, mu, logvar, z

