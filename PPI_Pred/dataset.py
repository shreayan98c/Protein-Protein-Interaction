import torch
from tdc.multi_pred import PPI


class HuRIDataset():
    def __init__(self, tokenizer, max_len=2048, data_split='train', neg_sample=2):
        self.data = PPI(name='HuRI')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data.neg_sample(frac=neg_sample)
        split = self.data.get_split()

        self.train = split["train"]  # each of these are a pd dataframe
        self.valid = split["valid"]
        self.test = split["test"]

        if data_split == 'train':
            self.data = self.train
        elif data_split == 'test':
            self.data = self.test
        else:
            self.data = self.valid

    def __getitem__(self, index):
        record = self.data.iloc[index]
        seq1 = self.tokenizer(record['Protein1'],
                              add_special_tokens=True, max_length=self.max_len, return_tensors="pt",
                              padding="max_length", truncation=False)
        seq2 = self.tokenizer(record['Protein2'],
                              add_special_tokens=True, max_length=self.max_len, return_tensors="pt",
                              padding="max_length", truncation=False)
        return seq1, seq2, torch.tensor(record['Y'])

    def __len__(self):
        return len(self.data)
