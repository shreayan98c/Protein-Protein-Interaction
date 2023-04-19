import pandas as pd
import seaborn as sns
from tdc.multi_pred import PPI
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

neg_sample = 2
threshold = 5000

data = PPI(name='HuRI')
data.neg_sample(frac=neg_sample)
split = data.get_split()

train = split["train"]  # each of these are a pd dataframe
valid = split["valid"]
test = split["test"]

train['seq1_length'] = train['Protein1'].str.len()
train['seq2_length'] = train['Protein2'].str.len()

# thresholding the protein seq lengths
train = train[train['seq1_length'] < threshold]
train = train[train['seq2_length'] < threshold]

print('Columns:', train.columns)
print('Shape:', train.shape)
print('Dist of target var:\n', train['Y'].value_counts())
print(train.head())

# distribution of protein lengths sequence lengths

# seq1
plt.hist(train['seq1_length'], bins=100)
plt.xlabel('Idx')
plt.ylabel('Protein 1 Length')
plt.show()

# seq2
plt.hist(train['seq2_length'], bins=100)
plt.xlabel('Idx')
plt.ylabel('Protein 2 Length')
plt.show()

# distribution of labels
sns.barplot(x=train['Y'].value_counts().index, y=train['Y'].value_counts())
plt.xlabel('Label Frequency')
plt.ylabel('Counts')
plt.show()
