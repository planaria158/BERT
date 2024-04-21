import os
import yaml
import argparse
import math
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

#----------------------------------------------------------------------
# This file is for training the BERT model on the FAB sequence data in 
# Masked Language Model mode. 
# currently training on dual GeForce RTX 2080 Ti with 11GB memory each
#----------------------------------------------------------------------
seed = 0
pl.seed_everything(seed)

def train(args):
    #
    # Read the config
    #
    config_path = './config/fab_sequence_data.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = config['model_params']
    print(config)


    class NewGELU(nn.Module):
        """
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
        """
        def forward(self, x):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    class CausalSelfAttention(nn.Module):
        """
        A vanilla multi-head masked self-attention layer with a projection at the end.
        It is possible to use torch.nn.MultiheadAttention here but I am including an
        explicit implementation here to show that there is nothing too scary here.
        """

        def __init__(self, config):
            super().__init__()
            assert config['n_embd'] % config['n_head'] == 0
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
            # output projection
            self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
            # regularization
            self.attn_dropout = nn.Dropout(config['attn_pdrop'])
            self.resid_dropout = nn.Dropout(config['resid_pdrop'])
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                        .view(1, 1, config['block_size'], config['block_size'])) # Not needed (GPT)
            self.n_head = config['n_head']
            self.n_embd = config['n_embd']
            self.block_size = config['block_size']

        def forward(self, x, mask):
            # mask (B (batch_size) x T (seq_len))

            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2) # BUG LEAKING???
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # GPT directional masking

            if mask is not None:
                att = att.masked_fill(mask[:, None, None, :] != 0, float('-inf')) # BERT-style masking

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))

            return y

    class Block(nn.Module):
        """ an unassuming Transformer block """

        def __init__(self, config):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config['n_embd'])
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config['n_embd'])
            self.mlp = nn.ModuleDict(dict(
                c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd']),
                c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd']),
                act     = NewGELU(),
                dropout = nn.Dropout(config['resid_pdrop']),
            ))
            m = self.mlp
            self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

        def forward(self, x, mask):
            x = x + self.attn(self.ln_1(x), mask)
            x = x + self.mlpf(self.ln_2(x))
            return x


    class BERT(nn.Module):
        """ BERT Language Model """

        def __init__(self, config):
            super().__init__()
            assert config['vocab_size'] is not None
            assert config['block_size'] is not None
            self.block_size = config['block_size']
            self.vocab_size = config['vocab_size']
            print('block_size:', self.block_size)
            print('vocab_size:', self.vocab_size)

            # type_given = config['model_type'] is not None
            # params_given = all([config['n_layer'] is not None, config['n_head'] is not None, config['n_embd'] is not None])
            # assert type_given ^ params_given # exactly one of these (XOR)
            # if type_given:
            #     # translate from model_type to detailed configuration
            #     config['merge_from_dict']({
            #         # names follow the huggingface naming conventions
            #         # GPT-1 yer=12, n_head=12, n_embd=768),  # 117M params
            #         # GPT-2 configs
            #         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            #         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            #         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            #         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
            #         # Gophers
            #         'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
            #         # (there are a number more...)
            #         # I made these tiny models up
            #         'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
            #         'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
            #         'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            #     }[config['model_type']])

            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config['vocab_size'], config['n_embd']),
                wpe = nn.Embedding(config['block_size'], config['n_embd']),
                drop = nn.Dropout(config['embd_pdrop']),
                h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
                ln_f = nn.LayerNorm(config['n_embd']),
            ))
            self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

            # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

            # report number of parameters (note we don't count the decoder parameters in lm_head)
            n_params = sum(p.numel() for p in self.transformer.parameters())
            print("number of parameters: %.2fM" % (n_params/1e6,))

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # @classmethod
        # def from_pretrained(cls, model_type):
        #     """
        #     Initialize a pretrained GPT model by copying over the weights
        #     from a huggingface/transformers checkpoint.
        #     """
        #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        #     from transformers import GPT2LMHeadModel

        #     # create a from-scratch initialized minGPT model
        #     config = cls.get_default_config()
        #     config['model_type'] = model_type
        #     config['vocab_size'] = 50257 # openai's model vocabulary
        #     config['block_size'] = 1024  # openai's model block_size
        #     model = BERT(config)
        #     sd = model.state_dict()

        #     # init a huggingface/transformers model
        #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        #     sd_hf = model_hf.state_dict()

        #     # copy while ensuring all of the parameters are aligned and match in names and shapes
        #     keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        #     # this means that we have to transpose these weights when we import them
        #     assert len(keys) == len(sd)
        #     for k in keys:
        #         if any(k.endswith(w) for w in transposed):
        #             # special treatment for the Conv1D weights we need to transpose
        #             assert sd_hf[k].shape[::-1] == sd[k].shape
        #             with torch.no_grad():
        #                 sd[k].copy_(sd_hf[k].t())
        #         else:
        #             # vanilla copy over the other parameters
        #             assert sd_hf[k].shape == sd[k].shape
        #             with torch.no_grad():
        #                 sd[k].copy_(sd_hf[k])

        #     return model

        # def configure_optimizers(self, train_config):
        #     """
        #     This long function is unfortunately doing something very simple and is being very defensive:
        #     We are separating out all parameters of the model into two buckets: those that will experience
        #     weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        #     We are then returning the PyTorch optimizer object.
        #     """

        #     # separate out all parameters to those that will and won't experience regularizing weight decay
        #     decay = set()
        #     no_decay = set()
        #     whitelist_weight_modules = (torch.nn.Linear, )
        #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        #     for mn, m in self.named_modules():
        #         for pn, p in m.named_parameters():
        #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        #             # random note: because named_modules and named_parameters are recursive
        #             # we will see the same tensors p many many times. but doing it this way
        #             # allows us to know which parent module any tensor p belongs to...
        #             if pn.endswith('bias'):
        #                 # all biases will not be decayed
        #                 no_decay.add(fpn)
        #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        #                 # weights of whitelist modules will be weight decayed
        #                 decay.add(fpn)
        #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        #                 # weights of blacklist modules will NOT be weight decayed
        #                 no_decay.add(fpn)

        #     # validate that we considered every parameter
        #     param_dict = {pn: p for pn, p in self.named_parameters()}
        #     inter_params = decay & no_decay
        #     union_params = decay | no_decay
        #     assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        #     assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        #                                                 % (str(param_dict.keys() - union_params), )

        #     # create the pytorch optimizer object
        #     optim_groups = [
        #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config['weight_decay']},
        #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        #     ]

        #     optimizer = torch.optim.AdamW(optim_groups, lr=train_config['learning_rate'], betas=train_config['betas'])
        #     return optimizer

        def forward(self, idx, mask=None):
            device = idx.device
            b, t = idx.size()
            assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

            # forward the GPT model itself
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x, mask)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)

            # if we are given some desired targets also calculate the loss
            idx = idx.view(-1)

            # Run in Masked Language Model mode
            if mask is not None:
                mask = mask.view(-1)
                mask_idx = torch.nonzero(mask)
                loss = F.cross_entropy(logits.view(-1, self.vocab_size),  mask, reduction='none')
                loss = loss.sum() / mask_idx.shape[0]
            else:
                loss = 0

            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, mask_token, temperature=1.0, do_sample=False, top_k=None):
            """
            Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            """
            device = idx.device

            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.block_size - 1 else idx[:, -self.block_size+1:]

                mask = torch.cat((torch.zeros_like(idx_cond).to(device), torch.tensor([[mask_token]]).to(device)), 1)
                idx_cond = torch.cat((idx_cond, torch.tensor([[mask_token]]).to(device)), 1)

                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond, mask)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # either sample from the distribution or take the most likely element

                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            return idx


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


    # train_data_path = './data/mit-ll/mit-ll-AlphaSeq_Antibody_Dataset-a8f64a9/antibody_dataset_2/train_set.pkl'
    # test_data_path = './data/mit-ll/mit-ll-AlphaSeq_Antibody_Dataset-a8f64a9/antibody_dataset_2/test_set.pkl'

    train_data_path = './data/mit-ll/mit-ll-AlphaSeq_Antibody_Dataset-a8f64a9/antibody_dataset_2/MITLL_AAlphaBio_Ab_Binding_dataset2.csv'
    train_dataset = FABSequenceDataset(config, train_data_path, skiprows=6)
    print(train_dataset.__len__())
    config['vocab_size'] = train_dataset.get_vocab_size()
    config['block_size'] = train_dataset.get_block_size()
    print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])

    # test_dataset = FABSequenceDataset(config, test_data_path)
    # print(test_dataset.__len__())
    # print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])


    # setup the dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=config['batch_size'], num_workers=config['num_workers'])
    # test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], num_workers=config['num_workers'])


    #----------------------------------------------------------
    # Pytorch Lightning Module that hosts the BERT model
    # and runs the training, validation, and testing loops
    #----------------------------------------------------------
    class BERT_Lightning(LightningModule):
        def __init__(self, config):
            super(BERT_Lightning, self).__init__()
            self.config = config
            self.model = BERT(config)
            self.criterion = nn.MSELoss()
            self.save_hyperparameters()

        def forward(self, x, mask):
            return self.model(x, mask)

        def common_forward(self, batch, batch_idx):
            x, mask = batch
            logits, loss = model(x, mask)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            return logits, loss
            # y_hat = torch.squeeze(y_hat)
            # loss = self.criterion(y_hat, y)
            # return loss, y_hat, y, transformer_out, attns, trans_input

        def training_step(self, batch, batch_idx):
            logits, loss = self.common_forward(batch, batch_idx)
            self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
            return loss

    
        # def on_validation_start(self):
        #     self.y_preds = []
        #     self.y_true = []
        #     self.metrics = None

        # def validation_step(self, batch, batch_idx):
        #     val_loss, y_hat, y, _, _, _ = self.common_forward(batch, batch_idx)
        #     self.y_true.extend(y.cpu().numpy().tolist())
        #     self.y_preds.extend(y_hat.cpu().numpy().tolist())
        #     self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        #     self.logger.experiment.add_scalars('loss', {'valid': val_loss}, self.global_step)
        #     return val_loss
        
        # def on_validation_end(self):
        #     assert(len(self.y_preds) == len(self.y_true))
        #     self.metrics = self._get_avm_metrics(self.y_true, self.y_preds)
        #     mape_metrics = {'MAPE':self.metrics['MAPE'], 'mdAPE':self.metrics['mdAPE']}
        #     ppe_metrics = {'PPE10':self.metrics['PPE10'], 'PPE20':self.metrics['PPE20']}
        #     self.logger.experiment.add_scalars('mape', mape_metrics, self.current_epoch)
        #     self.logger.experiment.add_scalars('ppe', ppe_metrics, self.current_epoch)
        #     self.y_preds = None
        #     self.y_true = None
        #     return 

        # def on_test_start(self):
        #     self.y_preds = []
        #     self.y_true = []
        #     self.metrics = None

        # def test_step(self, batch, batch_idx):
        #     test_loss, y_hat, y, transformer_out, attns, trans_input = self.common_forward(batch, batch_idx)
        #     self.log_dict({"test_loss": test_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        #     self.logger.experiment.add_scalars('loss', {'test': test_loss},self.global_step)
        #     self.y_true.extend(y.cpu().numpy().tolist())
        #     self.y_preds.extend(y_hat.cpu().numpy().tolist())
        #     return test_loss

        # def on_test_end(self):
        #     assert(len(self.y_preds) == len(self.y_true))
        #     self.metrics = self._get_avm_metrics(self.y_true, self.y_preds)
        #     print(self.metrics)
        #     return 
        
        # Function to get evaluation metrics (MAPE, MdAPE, PPE10, PPE20)
        # def _get_avm_metrics(self, y_true, y_pred):
        #     y_true = np.exp(np.array(y_true))  # convert back to house prices.
        #     y_pred = np.exp(np.array(y_pred))
        #     ppe10 = np.count_nonzero((np.abs((np.divide(y_pred, y_true)) - 1) <= 0.1)) / y_pred.size
        #     ppe20 = np.count_nonzero((np.abs((np.divide(y_pred, y_true)) - 1) <= 0.2)) / y_pred.size
        #     mape  = (np.sum(np.abs(y_true - y_pred) / y_true))/len(y_true)
        #     mdape = np.median(np.abs(y_true - y_pred)/y_true)

        #     return {'MAPE': mape, 'mdAPE': mdape,
        #             'PPE10': ppe10, 'PPE20': ppe20,
        #             'Sample_Size': y_true.shape[0]}


        def configure_optimizers(self):
            lr = self.config['learning_rate']
            optimizer = torch.optim.AdamW(self.model.parameters(), betas=config['betas'], lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_gamma'])
            return [optimizer], [scheduler]


    model = BERT_Lightning(config) 

    total_params = sum(param.numel() for param in model.parameters())
    print('Model has:', int(total_params), 'parameters')


    #--------------------------------------------------------------------
    # Training
    #--------------------------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=config['save_top_k'],
        every_n_train_steps=config['checkpoint_every_n_train_steps'],
        save_on_train_epoch_end=True,
        monitor = config['monitor'],
        mode = config['mode']
    )

    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=config['log_dir'], default_hp_metric=False)

    print('Using', config['accelerator'])
    trainer = pl.Trainer(strategy='ddp', #'ddp_find_unused_parameters_true', 
                            accelerator=config['accelerator'], 
                            devices=config['devices'],
                            max_epochs=config['num_epochs'],   
                            logger=logger, 
                            log_every_n_steps=config['log_every_nsteps'], 
                            callbacks=[checkpoint_callback])   


    trainer.fit(model=model, train_dataloaders=train_loader) #, val_dataloaders=test_loader) 



    # #--------------------------------------------------------------------
    # # LightningModule
    # #--------------------------------------------------------------------
    # print('Restarting from checkpoint')
    # path = os.path.join(train_config['log_dir'], train_config['checkpoint_name'])
    # model = TableTransformer_Lightning.load_from_checkpoint(checkpoint_path=path,
    #                                             model_config=model_config, 
    #                                             train_config=train_config,
    #                                             categories=categories,
    #                                             num_continuous=num_continuous)
    # print('Starting from new model instance')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for BERT training')
    parser.add_argument('--config', dest='config_path',
                        default='config/fab_sequence_data.yaml', type=str)
    args = parser.parse_args()
    train(args)


