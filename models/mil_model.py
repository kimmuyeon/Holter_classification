import torch.nn as nn
import torch

class SimpleMIL(nn.Module):
    def __init__(self, input_dim=140):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, bag):
        # bag: [N_inst, input_dim]
        h = self.feat(bag)               # [N, 32]
        m, _ = torch.max(h, dim=0)       # [32]  (max-pooling)
        out  = self.classifier(m)        # [1]
        return out

