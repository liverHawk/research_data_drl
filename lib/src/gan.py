import torch
import torch.nn as nn


class VAEConfig:
    def __init__(self):
        self.batch_size = 64

        self.beta1 = 0.5
        self.beta2 = 0.999
        
        self.epochs = 100
        self.le = 0.001
        self.kld_weight = 0.01

        self.input_dim = 784  # Example for MNIST
        self.hidden_dim = 400
        self.latent_dim = 20


class GANConfig:
    def __init__(self):
        self.batch_size = 64

        self.beta1 = 0.5
        self.beta2 = 0.999

        self.epochs = 100
        self.le = 0.0002


class Config:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.batch_size = 64
        self.vae = VAEConfig()
        self.gan = GANConfig()


class VAEEncoder(nn.Module):
    def __init__(self, config):
        super(VAEEncoder, self).__init__()
        self.config = config
        self.vae_config = config.vae

        self.fc1 = nn.Linear(self.vae_config.input_dim, self.vae_config.hidden_dim)
        self.fc2 = nn.Linear(self.vae_config.hidden_dim, self.vae_config.latent_dim)
        self.fc3 = nn.Linear(self.vae_config.hidden_dim, self.vae_config.latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        mu = self.fc3(h)
        logvar = self.fc3(h)
        return mu, logvar
    
    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    def __init__(self, config):
        super(VAEDecoder, self).__init__()
        self.config = config
        self.vae_config = config.vae

        self.fc1 = nn.Linear(self.vae_config.latent_dim, self.vae_config.hidden_dim)
        self.fc2 = nn.Linear(self.vae_config.hidden_dim, self.vae_config.hidden_dim * 2)
        self.fc3 = nn.Linear(self.vae_config.hidden_dim * 2, self.vae_config.input_dim)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)  # For BCEWithLogitsLoss, no sigmoid here


class AnomalyGenerator(nn.Module):
    def __init__(self, config):
        super(AnomalyGenerator, self).__init__()
        self.config = config
        self.gan_config = config.gan

        self.fc1 = nn.Linear(self.gan_config.input_dim, self.gan_config.hidden_dim)
        self.fc2 = nn.Linear(self.gan_config.hidden_dim, self.gan_config.hidden_dim)
        self.fc3 = nn.Linear(self.gan_config.hidden_dim, self.gan_config.input_dim)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.gan_config = config.gan

        self.fc1 = nn.Linear(self.gan_config.input_dim, self.gan_config.hidden_dim * 2)
        self.fc2 = nn.Linear(self.gan_config.hidden_dim * 2, self.gan_config.hidden_dim)
        self.fc3 = nn.Linear(self.gan_config.hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.sigmoid(self.fc3(h))