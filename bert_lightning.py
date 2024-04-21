import torch
from torch import nn
from pytorch_lightning.core import LightningModule
from bert_model import BERT

#--------------------------------------------------------
# Code fragments taken from:
# * https://github.com/barneyhill/minBERT
# * https://github.com/karpathy/minGPT

# protein sequence data taken from:
# * https://www.nature.com/articles/s41467-023-39022-2
# * https://zenodo.org/records/7783546
#--------------------------------------------------------

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
        logits, loss = self.model(x, mask)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm_clip'])
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
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=self.config['betas'], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config['lr_gamma'])
        return [optimizer], [scheduler]
