import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_layers: list = None, dropout: float = 0.3):
        super().__init__()

    def forward(self, input1, input2):
        raise NotImplementedError
