import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce, EinMix
import operator

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def TokenMixer(dim, expansion_factor: int , dropout: float):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        EinMix('b hw c -> b hw c_hid', weight_shape='c c_hid', bias_shape='c_hid',
           c=dim, c_hid=inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        EinMix('b hw c_hid -> b hw c', weight_shape='c_hid c', bias_shape='c',
           c=dim, c_hid=inner_dim),
        nn.Dropout(dropout)
    )

def TokenMixerOut(dim, expansion_factor: int, out_dim: int, dropout: float,):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        EinMix('b hw c -> b hw c_hid', weight_shape='c c_hid', bias_shape='c_hid',
           c=dim, c_hid=inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        EinMix('b hw c_hid -> b hw c_out', weight_shape='c_hid c_out', bias_shape='c_out',
           c_out=out_dim, c_hid=inner_dim),
        nn.Dropout(dropout)
    )


def ChannelMixer(dim, expansion_factor: int, dropout: float):
    n_hidden = int(dim * expansion_factor)
    return nn.Sequential(
        EinMix('b hw c -> b hid c', weight_shape='hw hid', bias_shape='hid', 
            hw=dim, hid=n_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        EinMix('b hid c -> b hw c', weight_shape='hid hw', bias_shape='hw',  
            hw=dim, hid=n_hidden),
        nn.Dropout(dropout),
    )

def ChannelMixerOut(dim, expansion_factor: int, out_dim: int, dropout: float):
    n_hidden = int(dim * expansion_factor)
    return nn.Sequential(
        EinMix('b hw c -> b hid c', weight_shape='hw hid', bias_shape='hid', 
            hw=dim, hid=n_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        EinMix('b hid c -> b hw_out c', weight_shape='hid hw_out', bias_shape='hw_out',  
            hw_out=out_dim, hid=n_hidden),
        nn.Dropout(dropout),
    )



def MLPMixer(*, channels, tokens, depth, num_classes, 
             expansion_factor_channel = 4, 
             expansion_factor_token = 0.5, 
             dropout = 0.):
    
    return nn.Sequential(
        Rearrange('b c t -> b t c'),
        *[nn.Sequential(
            PreNormResidual(channels, ChannelMixer(tokens, expansion_factor_channel, dropout, )),
            PreNormResidual(channels, TokenMixer(channels, expansion_factor_token, dropout, ))
        ) for _ in range(depth)],
        nn.LayerNorm(channels),
        Reduce('b t c -> b c', 'mean'),
        nn.Linear(channels, num_classes)
    )


def MLPMixerRegressor(*, channels, tokens, depth, pred_window, pred_feats, 
             expansion_factor_channel = 4, 
             expansion_factor_token = 0.5, 
             expansion_factor_first = 1, 
             dropout = 0.):
    
    return nn.Sequential(
        nn.Linear(in_features=tokens, out_features=tokens*expansion_factor_first, bias=True),
        Rearrange('b c t -> b t c'),
        *[nn.Sequential(
            PreNormResidual(channels, ChannelMixer(tokens*expansion_factor_first, expansion_factor_channel, dropout, )),
            PreNormResidual(channels, TokenMixer(channels, expansion_factor_token, dropout, ))
        ) for _ in range(depth)],
        ChannelMixerOut(tokens*expansion_factor_first, expansion_factor_channel, pred_feats, dropout, ),
        TokenMixerOut(channels, expansion_factor_token, pred_window, dropout, ),
        Rearrange('b t c -> b c t')
    )
