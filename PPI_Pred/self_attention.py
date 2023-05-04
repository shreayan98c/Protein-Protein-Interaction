import torch
import numpy as np
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


class SDP_Attention(nn.Module):
    """
    Scaled Dot-Product Attention Layer as proposed in Attention is All You Need
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

# REMOOOVVVVEEEE


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Self_Attention_Block(nn.Module):
    """
    Self-Attention block consisting of multi-head attention, norm,
    and feed forward layers as proposed in Attention is All You Need
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False, device=None, dtype=None):
        """ 
        Instantiation takes same args as torch.nn.MultiheadedAttention
        Args:
            embed_dims = 
            num_heads = the number of heads
            dd_dim = dimensions of the compressed forward neural net layer. 
        """
        super().__init__()

        # query, key, value calculations
        self.q_w = nn.Linear(embed_dim, embed_dim)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)

        # Block to pass input 1 through before passing to cross attention layer
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.attn = SDP_Attention(self.q_w, self.k_w, self.v_w)
        self.l_norm = nn.LayerNorm(embed_dim)

        # feed forward neural net
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim)

        self.out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        # Take out channel dimension
        input = torch.squeeze(input)

        # calculate query key value
        query = self.q_w(input1)
        key = self.k_w(input2)
        value = self.v_w(input2)

        # calculate attention out + residual connection and layer norm
        attn_out = self.MultiheadedAttention(query, key, value)[0]
        attn_out = self.l_norm(input1 + attn_out)

        # FF net followed by add and layer norm
        ff_out = self.ff(attn_out)
        ff_out = self.l_norm(ff_out + attn_out)

        return self.sigmoid(self.out(ff_out))
