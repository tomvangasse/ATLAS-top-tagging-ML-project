import torch.nn as nn

class TopTaggerMLP(nn.Module):  # inherits from nn.Module, the base PyTorch class for networks
    def __init__(self, n_features=15, hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        layers, width = [], n_features
      
        for _ in range(n_layers):  # adds the hidden layers
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
          
        layers.append(nn.Linear(width, 1))  # produces one number per jet, the logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1) # changes the shape from [batch,1] to [batch,0]
