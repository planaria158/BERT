import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from bert_model import BERT

#----------------------------------------------------------
# Pytorch Lightning Module that hosts the BERT model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class BERT_Lightning(LightningModule):
    def __init__(self, config, load_from_checkpoint=False):
        super(BERT_Lightning, self).__init__()
        self.config = config
        self.model = BERT(config, load_from_checkpoint)
        self.save_hyperparameters()

    def forward(self, x, mask):
        return self.model(x, mask)

    def common_forward(self, batch, batch_idx):
        x, mask = batch
        logits, loss, tform_out = self.model(x, mask)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.common_forward(batch, batch_idx)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, val_loss = self.common_forward(batch, batch_idx)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]
