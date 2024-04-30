import os
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from oas_dataset_v2 import OASSequenceDataset_v2 as dataset
from bert_lightning import BERT_Lightning

"""
    This file is for pre-training the BERT model on the aa sequence data  
    running in masked language model mode. 

    currently training on dual GeForce RTX 2080 Ti with 11GB memory each
"""
def train(args):

    # Read the config
    config_path = './config/pretrain_config.yaml'  
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config = config['model_params']
    pl.seed_everything(config['seed'])
    print(config)

    #----------------------------------------------------------
    # Load the dataset
    #----------------------------------------------------------
    train_dataset = dataset(config, config['train_data_path'])
    print(train_dataset.__len__())
    config['vocab_size'] = train_dataset.get_vocab_size()
    print('config[vocab_size]:', config['vocab_size'], ', config[block_size]:', config['block_size'])
    
    test_dataset = dataset(config, config['test_data_path'])
    print(test_dataset.__len__())
    
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=config['batch_size'], num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], num_workers=5)

    #----------------------------------------------------------
    # Model
    #----------------------------------------------------------
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
    trainer = pl.Trainer(strategy='ddp', 
                         accelerator=config['accelerator'], 
                         devices=config['devices'],
                         max_epochs=config['num_epochs'],   
                         logger=logger, 
                         log_every_n_steps=config['log_every_nsteps'], 
                         callbacks=[checkpoint_callback])   


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for pre-training BERT')
    parser.add_argument('--config', dest='config_path',
                        default='config/pretrain_config.yaml', type=str)
    args = parser.parse_args()
    train(args)


