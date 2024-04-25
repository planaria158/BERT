import torch
from torch import nn
from bert_lightning import BERT_Lightning

"""
    Layer class for the MLP
"""
class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, normalize=True, activation=True):
        super(Layer, self).__init__()
        self.normalize = normalize
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(in_dim) if normalize else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.activation(self.linear(self.norm(x))))
        return out

"""
    Vanilla MLP
    The regular mlp is a 3-layer mlp: 
        input layer (input_dim, 4*input_dim)
        hidden layer(4*input_dim, 2*input_dim)
        output layer(2*input_dim, 1)
"""
class MLP(nn.Module):
    def __init__(self, config, input_dim):
        super(MLP, self).__init__()
        print('Regression head is MLP')
        mlp_hidden_mults = (4, 2) # hardwired with values from TabTransformer paper

        hidden_dimensions = [input_dim * t for t in  mlp_hidden_mults]
        all_dimensions = [input_dim, *hidden_dimensions, 1]
        dims_pairs = list(zip(all_dimensions[:-1], all_dimensions[1:]))
        layers = []
        for ind, (in_dim, out_dim) in enumerate(dims_pairs):
            print('making mlp. in_dim, out_dim:', in_dim, out_dim)
            if ind >= (len(dims_pairs) - 1) :
                # For regression, the very last Layer has no dropout, normalization, and activation
                layer = Layer(in_dim, out_dim, normalize=False, activation=False)
            else:
                layer = Layer(in_dim, out_dim, config['regress_head_drop'])
            
            layers.append(layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x_in):
        return self.net(x_in) 
    


class fineBERT(nn.Module):
    """ fine-tuning version of BERT Language Model """
    
    def __init__(self, config):
        super().__init__()

        # first: Load the pre-trained BERT model
        assert config['checkpoint_pretrained'] is not None, 'checkpoint_pretrained is None'
        print('Loading pre-trained BERT model from:', config['checkpoint_pretrained'])
        path = config['checkpoint_pretrained']
        bert_model = BERT_Lightning.load_from_checkpoint(checkpoint_path=path, model_config=config)
        bert_model.freeze()

        # pre-trained BERT model
        self.bert = bert_model.model
        # for param in self.bert.parameters(): # freeze all bert weights
        #     param.requires_grad = False

        self.regression_head = MLP(config, config['n_embd'])
        self.regression_head.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, x_in):
        _, _, tform_out = self.bert(x_in, mask=None)
        x = tform_out[:, 0, :]  # pick off the CLS token from bert for regression
        logits = self.regression_head(x)
        return logits

