import torch
import torch.nn as nn
import torch.nn.functional as F
from PPI_Pred.losses import *
from PPI_Pred.CrossAttentionModel import *
from PPI_Pred.self_attention import SelfAttentionBlockSingleSequence


class CL_AttentionModel(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, seq_len):
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

        return cross_out_1, cross_out_2
    
class CL_AttentionModelClassification(nn.Module):
    def __init__(self, embed_dim, seq_len):
        from PPI_Pred.utils import LitContrastivePretrainer
        super(CL_AttentionModelClassification, self).__init__()

        pt_model = LitContrastivePretrainer.load_from_checkpoint("cl_attention_model.ckpt")
        self.pretrained_model = pt_model.model
        self.pretrained_model.eval()
        print('Loaded the pretrained model trained on Contrastive Loss')

        # Freeze the weights of the pretrained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.ff_out = nn.Linear(2 * embed_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform(self.ff_out.weight)
        self.fc1.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        
        out1, out2 = self.pretrained_model(input1, input2)

        # Flatten and concatenate outputs
        out1 = torch.flatten(out1, 1)
        out2 = torch.flatten(out2, 1)

        cat_output = torch.cat((out1, out2), 1)

        # Pass through final feed forward layer and activation
        ff_out = self.ff_out(cat_output)
        
        return self.sigmoid(ff_out)


class CL_Attention_ConvModel(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, conv_dim):
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

        # blocks of convolutional layers followed by batch normalization, relu, and max pooling
        self.conv1 = nn.Conv1d(in_channels=conv_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 62, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def forward_once(self, x):
        # convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        return x

    def forward(self, input1, input2):
        input1 = input1.permute(0, 2, 1)
        input2 = input2.permute(0, 2, 1)

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = output1.permute(0, 2, 1)
        output2 = output2.permute(0, 2, 1)

        # Pass each sequence through self attention
        self_out_1 = self.self_attention_1(output1)
        self_out_2 = self.self_attention_2(output2)

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(self_out_1, self_out_2)
        cross_out_2 = self.cross_attention_2(self_out_2, self_out_1)

        return cross_out_1, cross_out_2


class Conv_block(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, conv_dim):
        """
        Instantiation for the convolution blocks
        """
        super().__init__()

        # Blocks to pass both sequences through to compute self attention on them

        # Mix self attention scores
        self.cross_attention_1 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)
        self.cross_attention_2 = CrossAttentionBlock(embed_dim, num_heads, ff_dim, seq_len)

        self.self_attention_1 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)
        self.self_attention_2 = SelfAttentionBlockSingleSequence(embed_dim, num_heads, ff_dim, seq_len)

        # blocks of convolutional layers followed by batch normalization, relu, and max pooling
        self.conv1 = nn.Conv1d(in_channels=conv_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 62, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def forward_once(self, x):
        # convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        return x

class Attention_ConvModel(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, conv_dim):
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

        # blocks of convolutional layers followed by batch normalization, relu, and max pooling
        self.conv1 = nn.Conv1d(in_channels=conv_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 62, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def forward_once(self, x):
        # convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        return x

    def forward(self, input1, input2):
        input1 = input1.permute(0, 2, 1)
        input2 = input2.permute(0, 2, 1)

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output1 = output1.permute(0, 2, 1)
        output2 = output2.permute(0, 2, 1)

        # Pass each sequence through self attention
        self_out_1 = self.self_attention_1(output1)
        self_out_2 = self.self_attention_2(output2)

        # Pass sequences through cross attention
        cross_out_1 = self.cross_attention_1(self_out_1, self_out_2)
        cross_out_2 = self.cross_attention_2(self_out_2, self_out_1)

        

        return cross_out_1, cross_out_2

