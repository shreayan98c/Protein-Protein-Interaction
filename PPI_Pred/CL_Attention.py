import torch
import torch.nn as nn
import torch.nn.functional as F
from PPI_Pred.losses import *
from PPI_Pred.CrossAttentionModel import *

from PPI_Pred.self_attention import SelfAttentionBlockSingleSequence

class CL_AttentionModel(nn.Module):

    def __init__(self,embed_dim, num_heads, ff_dim, seq_len, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,\
        vdim=None, batch_first=False, device=None, dtype=None):

        """ 
        Instantiation takes same args as torch.nn.MultiheadedAttention
        Args:
            embed_dims = 
            num_heads = the number of heads
            dd_dim = dimensions of the compressed forward neural net layer. 
            seq_len = length of the sequence being passed in 
        """

        super().__init__()

        # Blocks to pass both sequences through to compute self attention on them

        # Mix self attention scores
        self.cross_attention_1 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)
        self.cross_attention_2 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)

        self.self_attention_1 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
        self.self_attention_2 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
    
    def forward(self, input1, input2):
        
        # Pass each sequence through self attention
        self_out_1 = self.self_attention_1(input1)
        self_out_2 = self.self_attention_2(input2)

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(self_out_1, self_out_2)
        cross_out_2 = self.cross_attention_2(self_out_2, self_out_1)

       
        return NULL