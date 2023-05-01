import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModel(nn.Module):
    def __init__(self, d: int 1):
        super().__init__()

        # Block to pass input 1 through before passing to cross attention layer
        block_input1 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Linear(in_features=64 * 625, out_features=128)
        )

    def forward(self, input1, input2):
        raise NotImplementedError
