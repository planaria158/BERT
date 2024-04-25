import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from fine_bert_model import fineBERT

#----------------------------------------------------------
# Pytorch Lightning Module that hosts the fineBERT model
# and runs the training, validation, and testing loops
#----------------------------------------------------------
class fineBERT_Lightning(LightningModule):
    def __init__(self, config):
        super(fineBERT_Lightning, self).__init__()
        self.config = config
        self.model = fineBERT(config)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def common_forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_forward(batch)
        self.log_dict({"loss": loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, _, _ = self.common_forward(batch)
        self.log_dict({"val_loss": val_loss}, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]
