import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SDP_Attention(nn.Module):
    """
    Scaled Dot-Product Attention Layer as proposed in Attention is All You Need"
    """

    def __init__(self, query, key, value, mask=None, dropout=None):
        """ 

        """
        super(SDP_Attention, self).__init__()
        self.dim = query.size(-1)
        self.query = query
        self.key = key
        self.value = value
        self.mask = mask
        self.dropout = dropout

    def forward(self):
        score = torch.matmul(self.query, self.key.transpose(-2, -1)) / torch.sqrt(self.dim)

        # Optional Masking (masks indicated values to -inf)
        # Fills elements of self tensor with value where mask is True. (-pytorch)
        if self.mask is not None:
            score = score.masked_fill(self.mask.view(score.size()), -float(np.inf))

        # Compute Scores
        attn = F.softmax(score, dim=-1)

        # Optional Dropout
        if self.dropout is not None:
            attn = self.dropout(attn)

        return torch.matmul(attn, self.value), attn
