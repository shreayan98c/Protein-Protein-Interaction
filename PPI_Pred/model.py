from PPI_Pred.losses import *


class MyModel(nn.Module):
    def __init__(self, levels: int = 3, blocks: int = 1, channels: int = 32, dropout: float = 0.5):
        super().__init__()

        self.encoder = nn.ModuleList()

        prev_channels = 1
        for l in range(levels):
            for _ in range(blocks):
                output_channels = channels * 2 ** l
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(prev_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                )
                )
                prev_channels = output_channels
            self.encoder.append(nn.MaxPool2d(2))
            self.encoder.append(nn.Dropout(dropout))

        self.encoder.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder.append(nn.Flatten())

        self.head = nn.Sequential(
            nn.Linear(prev_channels, 10),
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.head(x)
        return x


class SimpleLinearModel(nn.Module):
    def __init__(self, max_len, hidden_layers: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [50, 25, 3, 1]
        self.layers = nn.ModuleList()

        prev_channels = 2 * max_len
        for i, l in enumerate(hidden_layers[:-1]):
            self.layers.append(nn.Linear(prev_channels, l))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_channels = l

        self.layers.append(nn.Linear(prev_channels, 1))
        self.layers.append(nn.Sigmoid())

        # weight initialization
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, d: int = 500, pretrain: bool = False):
        super(SiameseNetwork, self).__init__()
        self.pretrain = pretrain

        # blocks of convolutional layers followed by batch normalization, relu, and max pooling
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=16, kernel_size=3, padding=1)
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
        self.fc1 = nn.Linear(in_features=64 * 40, out_features=128)
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

        # flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout2(x)

        return x

    def forward(self, input1, input2):

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.pretrain:
            return output1, output2

        else:
            diff = torch.abs(output1 - output2)
            output = self.sigmoid(self.fc3(diff))

            return output


class SiameseNetworkPretrainer(nn.Module):
    def __init__(self, d: int = 500):
        super(SiameseNetworkPretrainer, self).__init__()

        # blocks of convolutional layers followed by batch normalization, relu, and max pooling
        self.conv1 = nn.Conv1d(in_channels=d, out_channels=16, kernel_size=3, padding=1)
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
        self.fc1 = nn.Linear(in_features=64 * 40, out_features=128)
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

        # flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout2(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


class SiameseNetworkClassification(nn.Module):
    def __init__(self):
        from PPI_Pred.utils import LitContrastivePretrainer
        super(SiameseNetworkClassification, self).__init__()

        pt_model = LitContrastivePretrainer.load_from_checkpoint("siamese_pretrained.pt")
        self.pretrained_model = pt_model.model
        self.pretrained_model.eval()
        print('Loaded the pretrained model trained on Contrastive Loss')

        # Freeze the weights of the SiameseNet model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=64, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # weight initialization
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        output1, output2 = self.pretrained_model(input1, input2)
        diff = torch.abs(output1 - output2)
        output = self.dropout1(self.relu1(self.fc1(diff)))
        output = self.dropout2(self.relu2(self.fc2(output)))
        output = self.sigmoid(output)

        return output
