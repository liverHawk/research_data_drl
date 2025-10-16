import torch.nn as nn
import torch.nn.functional as F

class DeepFlowNetworkV2(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DeepFlowNetworkV2, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_outputs)

    def forward(self, x):
        renew = F.relu(self.fc1(x))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)