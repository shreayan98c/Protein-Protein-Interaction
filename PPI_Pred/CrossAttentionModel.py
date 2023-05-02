import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block with res connections, batch norm and a feed forward nn as defined in Attention is All You Need"
    """
    def __init__(self,embed_dim, kqv_dims, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,\
        vdim=None, batch_first=False, device=None, dtype=None):
        """ 
        Instantiation takes same args as torch.nn.MultiheadedAttention
        """
        super().__init__()

        # Block to pass input 1 through before passing to cross attention layer
        self.query = nn.Linear(embed_dim, kqv_dims)
        self.key = nn.Linear(embed_dim, kqv_dims)
        self.value = nn.Linear(embed_dim, kqv_dims)
        self.MultiheadedAttention = nn.MultiheadAttention(kqv_dims, num_heads)


    def forward(self, input1, input2):
        ### calculate k,q,v
        q = self.query(input1)
        k = self.key()
        v = self.value()

        ### calculate attention out
        attn_out = self.MultiheadedAttention(q,k,v)

        #calculate residual 
        residual = input1 + attn_out
