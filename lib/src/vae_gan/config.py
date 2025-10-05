import os
import torch
import yaml

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class ModelConfig:
    data_type: str = "image"  # "image", "vector"
    input_dim: int = 784  # Example for MNIST
    latent_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    use_spectral_norm: bool = False
    use_grad_penalty: bool = False
    grad_penalty_weight: float = 10.0


@dataclass
class TrainingConfig:
    batch_size: int = 128
    vae_epochs: int = 100
    gan_epochs: int = 200
    vae_lr: float = 1e-3
    gan_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    kld_weight: float = 1e-3
    reconstruction_weight: float = 1.0
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 0.1
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
    gradient_clip: float = 1.0
