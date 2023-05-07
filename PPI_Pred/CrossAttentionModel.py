import torch
import torch.nn as nn
import torch.nn.functional as F

from PPI_Pred.self_attention import SelfAttentionBlockSingleSequence


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

        # Block to pass input 1 through before passing to cross attention layer
        self.MultiheadedAttention = nn.MultiheadAttention(embed_dim,num_heads)
        self.l_norm = nn.LayerNorm(embed_dim)

        #query, key, value calculations
        self.q_w = nn.Linear(embed_dim, embed_dim)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)
    
        # feed forward neural net
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim)

        self.out = nn.Linear(embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input1, input2):

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
        return ff_out

class CrossAttentionModel(nn.Module):
    """
    Cross Attention Module with two cross attention blocks from both inputs
    """
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

        self.ff_out = nn.Linear(2 * embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(input1, input2)
        cross_out_2 = self.cross_attention_2(input2, input1)

        # Flatten and concatenate outputs
        cross_out_1 = torch.flatten(cross_out_1, 1)
        cross_out_2 = torch.flatten(cross_out_2, 1)

        cat_output = torch.cat((cross_out_1, cross_out_2), 1)

        # Pass through final feed forward layer and activation
        ff_out = self.ff_out(cat_output)
        
        return self.sigmoid(ff_out)
    
class SelfThenCrossAttentionModel(nn.Module):
    """
    Model that passes each sequence through a self attention block and then through a cross attention block
    """
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
        self.self_attention_1 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
        self.self_attention_2 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)

        # Mix self attention scores
        self.cross_attention_1 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)
        self.cross_attention_2 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)

        self.ff_out = nn.Linear(2 * embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        
        # Pass each sequence through self attention
        self_out_1 = self.self_attention_1(input1)
        self_out_2 = self.self_attention_2(input2)

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(self_out_1, self_out_2)
        cross_out_2 = self.cross_attention_2(self_out_2, self_out_1)

        # Flatten and concatenate outputs
        cross_out_1 = torch.flatten(cross_out_1, 1)
        cross_out_2 = torch.flatten(cross_out_2, 1)

        cat_output = torch.cat((cross_out_1, cross_out_2), 1)

        # Pass through final feed forward layer and activation
        ff_out = self.ff_out(cat_output)
        
        return self.sigmoid(ff_out)
    
class MultipleSelfThenCrossAttention(nn.Module):
    """
    Cross Attention Block with res connections, batch norm and a feed forward nn as defined in Attention is All You Need"
    """
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

        # Block to pass input 1 through before passing to cross attention layer
        self.MultiheadedAttention = nn.MultiheadAttention(embed_dim,num_heads)
        self.l_norm = nn.LayerNorm(embed_dim)

        #query, key, value calculations
        self.q_w = nn.Linear(embed_dim, embed_dim)
        self.k_w = nn.Linear(embed_dim, embed_dim)
        self.v_w = nn.Linear(embed_dim, embed_dim)
    
        # feed forward neural net
        self.ff = PositionwiseFeedForward(embed_dim, ff_dim)

        self.out = nn.Linear(embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input1, input2):

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
        return ff_out

class CrossAttentionModel(nn.Module):
    """
    Cross Attention Module with two cross attention blocks from both inputs
    """
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

        self.ff_out = nn.Linear(2 * embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(input1, input2)
        cross_out_2 = self.cross_attention_2(input2, input1)

        # Flatten and concatenate outputs
        cross_out_1 = torch.flatten(cross_out_1, 1)
        cross_out_2 = torch.flatten(cross_out_2, 1)

        cat_output = torch.cat((cross_out_1, cross_out_2), 1)

        # Pass through final feed forward layer and activation
        ff_out = self.ff_out(cat_output)
        
        return self.sigmoid(ff_out)
    
class MultipleSelfThenCrossAttentionModel(nn.Module):
    """
    Model that passes each sequence through 2 self attention block and then through 2 cross attention block
    """
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
        self.self_attention_11 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
        self.self_attention_21 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)

        self.self_attention_12 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
        self.self_attention_22 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)

        # Mix self attention scores
        self.cross_attention_1 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)
        self.cross_attention_2 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)

        self.ff_1 = nn.Linear(2 * embed_dim * seq_len, 100)
        self.relu = nn.ReLU()
        self.ff_out = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        
        # Pass each sequence through self attention
        self_out_1 = self.self_attention_11(input1)
        self_out_2 = self.self_attention_21(input2)

        self_out_1 = self.self_attention_12(self_out_1)
        self_out_2 = self.self_attention_22(self_out_2)

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(self_out_1, self_out_2)
        cross_out_2 = self.cross_attention_2(self_out_2, self_out_1)

        # Flatten and concatenate outputs
        cross_out_1 = torch.flatten(cross_out_1, 1)
        cross_out_2 = torch.flatten(cross_out_2, 1)

        cat_output = torch.cat((cross_out_1, cross_out_2), 1)

        # Pass through final feed forward layer and activation
        out = self.ff_1(cat_output)
        out = self.relu(out)
        out = self.ff_out(out)
        
        return self.sigmoid(out)