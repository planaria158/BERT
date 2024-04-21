import torch
from torch.utils.data import Dataset
import pandas as pd

#--------------------------------------------------------
# Code fragments taken from:
# * https://github.com/barneyhill/minBERT
# * https://github.com/karpathy/minGPT

# protein sequence data taken from:
# * https://www.nature.com/articles/s41467-023-39022-2
# * https://zenodo.org/records/7783546
#--------------------------------------------------------

class FABSequenceDataset(Dataset):
    """
    Emits batches of characters
    """
    def __init__(self, config, csv_file_path, skiprows):  #pk_file_path):
        self.config = config
        print('reading the data from:', csv_file_path)
        self.df = pd.read_csv(csv_file_path, skiprows=skiprows)
        # self.df = pk.load(open(pk_file_path, 'rb'))
        
        # my_set = set()   
        # def make_set(x):
        #     for c in x:
        #         my_set.add(c)

        # self.df['Sequence'].apply(make_set)
        # self.chars = sorted(list(my_set)) + ["[MASK]"]
        # print('len of chars:', len(self.chars))
        # print('chars:', self.chars)
    
        # 20 naturally occuring amino acids in human proteins plus MASK token
        self.chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '[MASK]']
        print('vocabulary:', self.chars)

        data_size, vocab_size = self.df.shape[0], len(self.chars)
        print('data has %d rows, %d vocab size (unique).' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return self.df.shape[0] #len(self.data) - self.config['block_size']

    """ Returns data, mask pairs used for Masked Language Model training """
    def __getitem__(self, idx):
        # grab a chunk of (block_size) characters from the data
        # chunk = self.data[idx:idx + self.config['block_size']]
        chunk = self.df.loc[idx, 'Sequence']
        
        # encode every character to an integer
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # get number of tokens to mask
        n_pred = max(1, int(round(self.config['block_size']*self.config['mask_prob'])))

        # indices of the tokens that will be masked (a random selection of n_pred of the tokens)
        masked_idx = torch.randperm(self.config['block_size'], dtype=torch.long, )[:n_pred]

        mask = torch.zeros_like(dix)

        # copy the actual tokens to the mask
        mask[masked_idx] = dix[masked_idx]
        
        # ... and overwrite then with MASK token in the data
        dix[masked_idx] = self.stoi["[MASK]"]

        return dix, mask 
