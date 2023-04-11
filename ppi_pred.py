# !pip install requirement.txt
from tdc.multi_pred import PPI

data = PPI(name='HuRI')
data.neg_sample(frac=2)
split = data.get_split()
train = split["train"]  # each of these are a pd dataframe
valid = split["valid"]
test = split["test"]
print(train.Y.value_counts())
print(train.shape)
print(valid.shape)
print(test.shape)
