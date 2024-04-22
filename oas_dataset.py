
import torch
from torch.utils.data import Dataset
import pickle as pk
import numpy as np

#--------------------------------------------------------
# Dataset for OAS data
#--------------------------------------------------------

class OASSequenceDataset(Dataset):
    """
    Emits sequences of aa's from the OAS data
    """
    def __init__(self, config, pk_file_path):
        super().__init__()
        self.config = config
        print('reading the data from:', pk_file_path)
        pk_data = pk.load(open(pk_file_path, 'rb'))
        self.data = list(pk_data)
    
        # 20 naturally occuring amino acids in human proteins plus MASK token
        # 'X' is a special token for unknown amino acids
        self.chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', '[MASK]']
        print('vocabulary:', self.chars)

        data_size, vocab_size = len(self.data), len(self.chars)
        print('data has %d rows, %d vocab size (unique).' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return len(self.data)

    """ Returns data, mask pairs used for Masked Language Model training """
    def __getitem__(self, idx):
        seq = self.data[idx]

        # get a randomly located block_size substring from the sequence
        if len(seq) == self.config['block_size']:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - self.config['block_size'])
            chunk = seq[start_idx:start_idx + self.config['block_size']]
        
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
