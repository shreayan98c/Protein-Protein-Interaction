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
    def __init__(self, hidden_layers: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [50, 25, 3, 1]
        self.layers = nn.ModuleList()

        prev_channels = 10000
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
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, d: int = 1):
        super(SiameseNetwork, self).__init__()

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
        self.fc1 = nn.Linear(in_features=64 * 625, out_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.bn5 = nn.BatchNorm1d(num_features=64)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(in_features=64, out_features=1)

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

        # calculate Euclidean distance between outputs
        # distance = F.pairwise_distance(output1, output2)
        # distance = distance.view(-1, 1)
        #
        # print(distance.shape)
        #
        # # pass through final fully connected layer
        # # out = self.fc3(distance)
        #
        # out = F.sigmoid(self.fc3(distance))
        #
        # print(out.shape)

        # return out


# create a siamese network
# References:
# https://builtin.com/machine-learning/siamese-network
# https://www.projectpro.io/article/siamese-neural-networks/718
# https://www.kaggle.com/parthdhameliya77/siamese-network-approach-82-acc
class SigNetSiameseNetwork(nn.Module):
    def __init__(self):
        super(SigNetSiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
