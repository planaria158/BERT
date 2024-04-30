
import torch
from torch.utils.data import Dataset
import pickle as pk
import numpy as np
import random
import math

#--------------------------------------------------------
# Dataset for OAS data
#--------------------------------------------------------
class OASSequenceDataset_v2(Dataset):
    """
    Emits sequences of aa's from the OAS data
    """
    def __init__(self, config, pk_file_path):
        super().__init__()
        self.config = config
        print('reading the data from:', pk_file_path)
        pk_data = pk.load(open(pk_file_path, 'rb'))
        self.data = list(pk_data)
    
        # 20 naturally occuring amino acids in human proteins plus MASK token, 
        # 'X' is a special token for unknown amino acids, CLS token is for classification, PAD for padding, and SEP for seperator
        self.chars = ['CLS', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', 'MASK', 'PAD', 'SEP']
        self.first_aa_idx = 1
        self.last_aa_idx = len(self.chars) - 4

        print('vocabulary:', self.chars)

        data_size, vocab_size = len(self.data), len(self.chars)
        print('data has %d rows, %d vocab size (unique).' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.vocab_size = vocab_size

    def get_random_aa_tokens(self, length):
        return torch.tensor([self.stoi[aa] for aa in random.choices(self.chars[self.first_aa_idx:self.last_aa_idx+1], k=length)], dtype=torch.long)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config['block_size']

    def __len__(self):
        return len(self.data)
    
    def __get_seq_mask_pair(self, idx, block_size):
        seq = self.data[idx]

        # get a randomly located block_size-1 substring from the sequence
        # '-1' so we can prepend the CLS token to the start of the encoded string
        if len(seq) <= block_size-1:
            chunk = seq
        else:
            start_idx = np.random.randint(0, len(seq) - (block_size - 1))
            chunk = seq[start_idx:(start_idx + block_size-1)]

        # encode every character to its token
        dix = torch.tensor([self.stoi[s] for s in chunk], dtype=torch.long)

        # prepend the CLS token to the sequence
        dix = torch.cat((torch.tensor([self.stoi['CLS']], dtype=torch.long), dix))

        first_aa = 1 # first aa position in the sequence (after CLS)
        last_aa = dix.shape[0] # last aa position in the sequence

        # dix now looks like: [[CLS], x1, x2, x3, ..., xN, [PAD], [PAD], ..., [PAD]]
        # Only mask aa tokens (never CLS, SEP, PAD)

        # get number of tokens to mask
        n_pred = max(1, int(round((last_aa - first_aa)*self.config['mask_prob'])))

        # indices of the tokens that will be masked (a random selection of n_pred of the tokens)
        masked_idx = torch.randperm(last_aa-1, dtype=torch.long, )[:n_pred]
        masked_idx += 1  # so we never mask the CLS token

        mask = torch.zeros_like(dix)

        # copy the actual tokens to the mask...
        mask[masked_idx] = dix[masked_idx]
        
        # ... and overwrite them in the data
        # with 80% probability, change to MASK token
        # with 10% probability, change to random aa token
        # with 10% probability, keep the same token
        p8 = int(math.ceil(n_pred*0.8))
        p9 = int(math.ceil(n_pred*0.9))
        mask_token_idxs = masked_idx[0:p8]
        rand_aa_idxs = masked_idx[p8:p9]
        keep_idxs = masked_idx[p9:]
        assert(masked_idx.shape[0] == mask_token_idxs.shape[0] + rand_aa_idxs.shape[0] + keep_idxs.shape[0])

        dix[mask_token_idxs] = self.stoi['MASK']    
        dix[rand_aa_idxs] = self.get_random_aa_tokens(rand_aa_idxs.shape[0])

        return dix, mask    
    


    """ Returns data, mask pairs used for Masked Language Model training """
    def __getitem__(self, idx):
        seq = self.data[idx]

        block_size = self.config['block_size']
        dix1, mask1 = self.__get_seq_mask_pair(idx, block_size)
        block_size -= dix1.shape[0] - 1 # -1 is to account for a SEP token we'll insert below
        jdx = random.randint(0, len(self.data) - 1)
        dix2, mask2 = self.__get_seq_mask_pair(jdx, block_size)

        # print('dix1 shape:', dix1.shape, ', dix2 shape:', dix2.shape)

        dix = torch.cat((dix1, torch.tensor([self.stoi['SEP']], dtype=torch.long), dix2[1:]))    # note: '[1:]' strips off ...
        mask = torch.cat((mask1, torch.tensor([self.stoi['SEP']], dtype=torch.long), mask2[1:])) # ... the CLS token in dix2

        # print('config[block_size]:', self.config['block_size'], ', dix:', dix.shape, ', mask:', mask.shape)

        # pad the end with PAD tokens if necessary
        if dix.shape[0] < self.config['block_size']:
            dix = torch.cat((dix, torch.tensor([self.stoi['PAD']] * (self.config['block_size'] - len(dix)), dtype=torch.long)))
            mask = torch.cat((mask, torch.tensor([0] * (self.config['block_size'] - len(mask)), dtype=torch.long)))
                             
        # dix now looks like: [[CLS], xa1, xa2, xa3, ..., xaN, [SEP], xb1, xb2, xb3, xbN...[PAD] ..., [PAD]]

        return dix, mask 
