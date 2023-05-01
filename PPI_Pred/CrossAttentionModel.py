import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block with res connections, batch norm and a feed forward nn as defined in Attention is All You Need"
    """
    def __init__(self,embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,\
        vdim=None, batch_first=False, device=None, dtype=None):
        """ 
        Instantiation takes same args as torch.nn.MultiheadedAttention
        """
        super().__init__()

        # Block to pass input 1 through before passing to cross attention layer
        self.query = nn.Linear()
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.MultiheadedAttention = nn.MultiheadAttention(embed_dim, num_heads)
        

    def forward(self, input1, input2):
        
