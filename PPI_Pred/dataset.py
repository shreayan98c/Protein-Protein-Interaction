import torch
from tdc.multi_pred import PPI
from torch.utils.data import Dataset


class HuRIDataset(torch.utils.data.Dataset):
    """
    Dataset class for HuRI dataset to be used with PyTorch DataLoader.
    """
    def __init__(self, tokenizer, model, small_subset, max_len=500, data_split='train', thresh=500, neg_sample=1):
        """
        Constructor for HuRIDataset class.
        :param tokenizer: tokenizer to use for encoding sequences
        :param model: model to use for embedding generation
        :param small_subset: subset of data to use for training
        :param max_len: max length of sequence
        :param data_split: train, test, or valid
        :param thresh: length threshold for sequences
        :param neg_sample: ratio of negative to positive samples
        """
        self.data = PPI(name='HuRI')
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len
        self.thresh = thresh
        self.data.neg_sample(frac=neg_sample)
        split = self.data.get_split()

        self.train = split["train"].sample(frac=1, random_state=1)
        self.valid = split["valid"].sample(frac=1, random_state=1)
        self.test = split["test"].sample(frac=1, random_state=1)

        # select subset of data if small_subset is True
        if small_subset:
            self.train = self.train[:1000]
            self.valid = self.valid[:1000]
            self.test = self.test[:1000]
            # self.train = self.train[:3]
            # self.valid = self.valid[:10]
            # self.test = self.test[:10]

        # find length of each sequence
        self.train['seq1_length'] = self.train['Protein1'].str.len()
        self.train['seq2_length'] = self.train['Protein2'].str.len()
        self.test['seq1_length'] = self.test['Protein1'].str.len()
        self.test['seq2_length'] = self.test['Protein2'].str.len()
        self.valid['seq1_length'] = self.valid['Protein1'].str.len()
        self.valid['seq2_length'] = self.valid['Protein2'].str.len()

        # apply threshold to all splits
        self.train = self.train[self.train['seq1_length'] < self.thresh].reset_index(drop=True)
        self.train = self.train[self.train['seq2_length'] < self.thresh].reset_index(drop=True)
        self.test = self.test[self.test['seq1_length'] < self.thresh].reset_index(drop=True)
        self.test = self.test[self.test['seq2_length'] < self.thresh].reset_index(drop=True)
        self.valid = self.valid[self.valid['seq1_length'] < self.thresh].reset_index(drop=True)
        self.valid = self.valid[self.valid['seq2_length'] < self.thresh].reset_index(drop=True)

        if data_split == 'train':
            self.data = self.train
            # self.data.to_csv('train.csv')
        elif data_split == 'test':
            self.data = self.test
            # self.data.to_csv('test.csv')
        else:
            self.data = self.valid
            # self.data.to_csv('valid.csv')

    def __getitem__(self, index):
        """
        Get item at index.
        :param index: index of item to get
        :return: dictionary of input ids, attention masks, and labels for both proteins and concatenated proteins
        """
        record = self.data.iloc[index]

        # encode sequences
        seq1 = self.tokenizer(record['Protein1'],
                              add_special_tokens=True, max_length=self.max_len, return_tensors="pt",
                              padding="max_length", truncation=True)
        seq1_encoded = self.model(**seq1)
        seq1_encoded = seq1_encoded.last_hidden_state
        seq1_encoded = torch.squeeze(seq1_encoded)

        seq2 = self.tokenizer(record['Protein2'],
                              add_special_tokens=True, max_length=self.max_len, return_tensors="pt",
                              padding="max_length", truncation=True)
        seq2_encoded = self.model(**seq2)
        seq2_encoded = seq2_encoded.last_hidden_state
        seq2_encoded = torch.squeeze(seq2_encoded)

        return {
            'seq1_encoded': seq2_encoded,
            'seq2_encoded': seq2_encoded,
            'concatenated_inputs': torch.cat((seq1_encoded, seq2_encoded), dim=1),
            'diff_inputs': torch.abs(seq1_encoded - seq2_encoded),
            'label': torch.tensor(record['Y'])
        }

    def __len__(self):
        """
        Get length of dataset.
        :return: length of dataset
        """
        return len(self.data)
