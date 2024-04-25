import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

#--------------------------------------------------------
# Code fragments taken from:
# * https://github.com/barneyhill/minBERT
# * https://github.com/karpathy/minGPT

# protein sequence data taken from:
# * https://www.nature.com/articles/s41467-023-39022-2
# * https://zenodo.org/records/7783546
#--------------------------------------------------------

class scFv_Dataset(Dataset):
    """
    Emits batches of amino acid sequences and binding energies
    """
    def __init__(self, config, csv_file_path, skiprows=0):  
        super().__init__()
        self.config = config
        print('reading the data from:', csv_file_path)
        self.df = pd.read_csv(csv_file_path, skiprows=skiprows)
        
        # 20 naturally occuring amino acids in human proteins plus MASK token, 
        # 'X' is a special token for unknown amino acids, and CLS token is for classification
        self.chars = ['CLS', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '[MASK]']
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
        seq = self.df.loc[idx, 'Sequence']
        affinity = self.df.loc[idx, 'Pred_affinity']
        assert not math.isnan(affinity), 'affinity is nan'
        assert affinity >= 0.0, 'affinity is negative'
        assert len(seq) >= self.config['block_size'], 'sequence is too short'

        # get a randomly located block_size-1 substring from the sequence
        # '-1' so we can prepend the CLS token to the start of the encoded string
        if len(seq) == self.config['block_size']-1:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (self.config['block_size'] - 1))
            chunk = seq[start_idx:start_idx + self.config['block_size']-1]

        # encode every character to an integer
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # prepend the CLS token to the sequence
        dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))

        mask = None
        if self.config['mask_prob'] > 0:
            # get number of tokens to mask
            n_pred = max(1, int(round(self.config['block_size']*self.config['mask_prob'])))

            # indices of the tokens that will be masked (a random selection of n_pred of the tokens)
            masked_idx = torch.randperm(self.config['block_size']-1, dtype=torch.long, )[:n_pred]
            masked_idx += 1  # so we never mask the CLS token

            mask = torch.zeros_like(dix)

            # copy the actual tokens to the mask
            mask[masked_idx] = dix[masked_idx]
            
            # ... and overwrite them with MASK token in the data
            dix[masked_idx] = self.stoi["[MASK]"]

        return dix, torch.tensor([affinity], dtype=torch.float32) 
