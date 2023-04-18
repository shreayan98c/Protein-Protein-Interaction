import numpy
import torch
from tdc.multi_pred import PPI
from torchvision.datasets import MNIST


class HuRIDataset():
    def __init__(self, type='train', neg_sample=2):
        self.data = PPI(name='HuRI')
        self.data.neg_sample(frac=neg_sample)
        split = self.data.get_split()

        self.train = split["train"]  # each of these are a pd dataframe
        self.valid = split["valid"]
        self.test = split["test"]

        if type == 'train':
            self.data = self.train
        elif type == 'test':
            self.data = self.test
        else:
            self.data = self.valid

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataset():
    def __init__(self, train=True):
        self.data = MNIST(root="data", train=train, download=True)

    def __getitem__(self, index):
        img, target = self.data[index]
        img = numpy.array(img)
        img = img / 255.0
        img = img.astype(numpy.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        target = torch.tensor(target, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.data)
