import torch
from tdc.multi_pred import PPI


class HuRIDataset():
    def __init__(self, tokenizer, max_len=5000, data_split='train', thresh=5000, neg_sample=2):
        self.data = PPI(name='HuRI')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.thresh = thresh
        self.data.neg_sample(frac=neg_sample)
        split = self.data.get_split()

        self.train = split["train"]
        self.valid = split["valid"]
        self.test = split["test"]

        # find length of each sequence
        self.train['seq1_length'] = self.train['Protein1'].str.len()
        self.train['seq2_length'] = self.train['Protein2'].str.len()
        self.test['seq1_length'] = self.test['Protein1'].str.len()
        self.test['seq2_length'] = self.test['Protein2'].str.len()
        self.valid['seq1_length'] = self.valid['Protein1'].str.len()
        self.valid['seq2_length'] = self.valid['Protein2'].str.len()

        # apply threshold to all splits
        self.train = self.train[self.train['seq1_length'] < self.thresh]
        self.train = self.train[self.train['seq2_length'] < self.thresh]
        self.test = self.test[self.test['seq1_length'] < self.thresh]
        self.test = self.test[self.test['seq2_length'] < self.thresh]
        self.valid = self.valid[self.valid['seq1_length'] < self.thresh]
        self.valid = self.valid[self.valid['seq2_length'] < self.thresh]

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
                              truncation=False)
        seq2 = self.tokenizer(record['Protein2'],
                              add_special_tokens=True, max_length=self.max_len, return_tensors="pt",
                              padding="max_length", truncation=False)
        print(seq1['input_ids'][0].shape)
        print(seq1['attention_mask'][0].shape)
        print(seq2['input_ids'][0].shape)
        print(seq2['attention_mask'][0].shape)
        return {
            'seq1_input_ids': seq1['input_ids'][0],
            'seq1_attention_mask': seq1['attention_mask'][0],
            'seq2_input_ids': seq2['input_ids'][0],
            'seq2_attention_mask': seq2['attention_mask'][0],
            'label': torch.tensor(record['Y'])
        }

    def __len__(self):
        return len(self.data)
