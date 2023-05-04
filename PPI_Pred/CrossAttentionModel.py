import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation, from the Annotated Transformer."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class CrossAttentionBlock(nn.Module):
    """
    Cross Attention Block with res connections, batch norm and a feed forward nn as defined in Attention is All You Need"
    """
    def __init__(self,embed_dim, num_heads, ff_dim, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,\
        vdim=None, batch_first=False, device=None, dtype=None):
        """ 
        Instantiation takes same args as torch.nn.MultiheadedAttention
        Args:
            embed_dims = 
            num_heads = the number of heads
            dd_dim = dimensions of the compressed forward neural net layer. 
        """
        super().__init__()

        # Block to pass input 1 through before passing to cross attention layer
        self.MultiheadedAttention = nn.MultiheadAttention(embed_dim,num_heads)
        self.l_norm = nn.LayerNorm(embed_dim)

        #query, key, value calculations
        self.q_w = nn.Linear(embed_dim, embed_dim)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)
    
        # feed forward neural net
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim)

        self.out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input1, input2):
        
        # Take out channel dimension
        input1 = torch.squeeze(input1)
        input2 = torch.squeeze(input2)

        ### calculate query key value
        query = self.q_w(input1)
        key = self.k_w(input2)
        value = self.v_w(input2)

        ### calculate attention out + residual connection and layer norm
        attn_out = self.MultiheadedAttention(query,key,value)[0]
        attn_out = self.l_norm(input1 + attn_out)

        ### FF net followed by add and layer norm
        ff_out = self.ff(attn_out)
        ff_out = self.l_norm(ff_out + attn_out)

        return self.sigmoid(self.out(ff_out))






