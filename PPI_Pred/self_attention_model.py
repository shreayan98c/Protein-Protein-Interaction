import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_Attn(nn.Module):
    """
    Self-Attention Layer as defined in Attention is All You Need"
    """
    def __init__(query, key, value, mask=None, dropout=None):
        """ 

        """
        super().__init__()

    def forward(self, sequence):
        pass
