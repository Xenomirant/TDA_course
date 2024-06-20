import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce, EinMix


class Conv_net(nn.Module):
    def __init__(self, classes: int, channels: int, lin_dim: int, conv_dim: int, dropout: int):
        super().__init__()

        self.model = nn.Sequential(
            Rearrange("b c f -> b f c"),
            nn.Conv1d(channels, conv_dim, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),
            nn.ReLU(),

            nn.Conv1d(conv_dim, conv_dim*3, kernel_size=5, stride=2, padding=1),
            nn.MaxPool1d(2),
            nn.Dropout1d(dropout),
            nn.ReLU(),

            Rearrange("b c t -> b (c t)"),

            nn.LazyLinear(lin_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lin_dim, classes), 
        )

    def forward(self, x):
        return self.model(x)


class DeepSets(torch.nn.Module):
    def __init__(self, n_in, n_hidden_enc, n_out_enc, n_hidden_dec=16, n_out_dec=5, dropout=.1):
        super(DeepSets, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc)
        self.decoder = MLP(n_out_enc, n_hidden_dec, n_out_dec)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, X):
        z_enc = self.dropout(self.encoder(X))
        z = self.decoder(z_enc)
        return z
    
class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(n_in, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, n_out)
        
    def forward(self, X):
        X = torch.nn.functional.relu(self.linear1(X))
        X = self.linear2(X)
        return X
    
class Encoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Encoder, self).__init__()
        self.mlp = MLP(n_in, n_hidden, n_out)
        self.layer_norm = torch.nn.LayerNorm(n_out)
        
    def forward(self, X):
        X = self.layer_norm(self.mlp(X))
        x = X.mean(dim=1) # aggregation
        return x