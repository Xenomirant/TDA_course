import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce, EinMix
import math
import operator


def TokenMixerOut(dim, out_dim: int, dropout: float, expansion_factor: int=2):
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

def ChannelMixerOut(dim, out_dim: int, dropout: float, expansion_factor: int=2):
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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = int(d_model / self.num_heads)
        self.w_qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.w_out = torch.nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_qkv.weight)
        self.w_qkv.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_out.weight)
        self.w_out.bias.data.fill_(0)

    def forward(self, x, mask=None):
        # Project into query, key, and value space in one shot.
        qkv = self.w_qkv(x)
        # Split into different heads.
        qkv = einops.rearrange(qkv,
                               'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                               num_heads=self.num_heads,
                               head_dim=3 * self.d_head
                               )

        # For each head split back into query, key, and value.
        q, k, v = einops.rearrange(qkv,
                                   'batch num_heads seq_len (split head_dim) -> split batch num_heads seq_len head_dim',
                                   split=3)

        # Reshape so we can do dot product with query. -1 dimension of query vector needs to match -2 dimension of key.
        k = einops.rearrange(k,
                             'batch num_heads seq_len head_dim -> batch num_heads head_dim seq_len')

        attention_logits = torch.matmul(q, k) / math.sqrt(q.size()[-1])
        if mask is not None:
            # Reshape so it work with the logits matrix
            mask = einops.rearrange(mask,
                                    'batch seq_len -> batch 1 1 seq_len')
            attention_logits = attention_logits.masked_fill(mask == 0, -9e15)
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)

        attention = torch.matmul(attention_weights, v)
        # merge the heads.
        attention = einops.rearrange(attention,
                                     'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')

        return x + self.w_out(attention)
                #attention_weights

class L2MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, **kwargs):
        super(L2MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = int(d_model / self.num_heads)
        # we only need two matrices as Q and K are tied for Lipschitzness
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_out = torch.nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_qkv.weight)
        self.w_qkv.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_out.weight)
        self.w_out.bias.data.fill_(0)

    def forward(self, x, mask=None):
        # Project into query, key, and value space in one shot.
        q = self.w_q(x)
        # Split into different heads.
        q = einops.rearrange(q,
                               'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                               num_heads=self.num_heads,
                               head_dim=self.d_head
                               )
        v = einops.rearrange(self.w_v.weight,
                               'd_model (num_heads head_dim) -> num_heads d_model head_dim',
                               num_heads=self.num_heads,
                               head_dim=self.d_head
                               )

        # Reshape so we can do dot product with query. -1 dimension of query vector needs to match -2 dimension of key.
        k = einops.rearrange(q,
                             'batch num_heads seq_len head_dim -> batch num_heads head_dim seq_len')

        W_q_T = einops.rearrange(self.w_q.weight, " d_model (num_heads head_dim) -> num_heads head_dim d_model", 
                                num_heads=self.num_heads,
                                head_dim=self.d_head
                                ) 
        
        # new regularized attention logits
        attention_logits = - (torch.sum(q**2, -1, keepdim=True) \
                              - 2*torch.matmul(q, k) + torch.sum(k**2, -2, keepdim=True)
                             ) / math.sqrt(q.size()[-1])
        if mask is not None:
            # Reshape so it work with the logits matrix
            mask = einops.rearrange(mask,
                                    'batch seq_len -> batch 1 1 seq_len')
            attention_logits = attention_logits.masked_fill(mask == 0, -9e15)
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)

        attention = attention_weights @ q @ W_q_T @ v / math.sqrt(q.size()[-1])
        # merge the heads.
        attention = einops.rearrange(attention,
                                     'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')

        return x + self.w_out(attention)
                #attention_weights


class KNNMultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, top_k: int):
        super(KNNMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.top_k = top_k
        self.d_head = int(d_model / self.num_heads)
        self.w_qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.w_out = torch.nn.Linear(d_model, d_model)
        

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_qkv.weight)
        self.w_qkv.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_out.weight)
        self.w_out.bias.data.fill_(0)

    def forward(self, x, mask=None):
        # Project into query, key, and value space in one shot.
        qkv = self.w_qkv(x)
        # Split into different heads.
        qkv = einops.rearrange(qkv,
                               'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                               num_heads=self.num_heads,
                               head_dim=3 * self.d_head
                               )

        # For each head split back into query, key, and value.
        q, k, v = einops.rearrange(qkv,
                                   'batch num_heads seq_len (split head_dim) -> split batch num_heads seq_len head_dim',
                                   split=3)

        # Reshape so we can do dot product with query. -1 dimension of query vector needs to match -2 dimension of key.
        k = einops.rearrange(k,
                             'batch num_heads seq_len head_dim -> batch num_heads head_dim seq_len')

        attention_logits = torch.matmul(q, k) / math.sqrt(q.size()[-1])

        knn_mask = torch.zeros(k.size()[0], self.num_heads, k.size()[-1], k.size()[-1], device=x.device, requires_grad=False)
        # get indices of top_k values along the row axis
        index = torch.topk(attention_logits, k=self.top_k, dim=-1, largest=True, sorted=False)[1]
        # fill the mask with ones at given index positions (like self.weights[i][j][k][index[i][j][k][l]] = 1.
        knn_mask.scatter_(-1, index, 1.)
        # fill dropped positions with -inf much like in masked attention
        attention_logits=torch.where(knn_mask>0, attention_logits, torch.full_like(attention_logits, float("-inf")))
        if mask is not None:
            # Reshape so it work with the logits matrix
            mask = einops.rearrange(mask,
                                    'batch seq_len -> batch 1 1 seq_len')
            attention_logits = attention_logits.masked_fill(mask == 0, -9e15)
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)

        attention = torch.matmul(attention_weights, v)
        # merge the heads.
        attention = einops.rearrange(attention,
                                     'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')

        return x + self.w_out(attention)
                #attention_weights


def MLP(n_in: int, n_hidden: int, n_out: int, activation, dropout: int=.1):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(n_in),
        torch.nn.Linear(n_in, n_hidden),
        activation(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(n_hidden, n_out)
    )

# MLP with residual connection
class ResMLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, activation, dropout=.1):
        super(ResMLP, self).__init__()
        self.mlp = MLP(n_in, n_hidden, n_out, activation, dropout)
    
    def forward(self, X):
        return X + self.mlp(X)

def AttentionBlock(num_heads: int, in_feats: int, feedforward_dim: int, activation, attention_type, dropout=.1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LayerNorm(in_feats),
        attention_type(num_heads=num_heads, d_model=in_feats, **kwargs),
        ResMLP(n_in=in_feats, n_hidden=feedforward_dim, n_out=in_feats, 
               activation=activation, dropout=dropout)
    )
    
class Encoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out, n_head, feedforward_dim, activation, n_layers = 1,  dropout = .1, **kwargs):
        super(Encoder, self).__init__()
        self.embedder = torch.nn.Sequential(
             MLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out, activation=activation, dropout=dropout,),
            *[AttentionBlock(num_heads=n_head, in_feats=n_out, feedforward_dim=feedforward_dim,
                          activation=activation, **kwargs) for _ in range(n_layers)],
        )
    def forward(self, X):
        return self.embedder(X)

class PersTransformer(torch.nn.Module):
    def __init__(self, n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, num_transformer_layers, 
                 n_hidden_dec=16, n_out_dec=5, dropout=.1, **kwargs):
        super(PersTransformer, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, 
                               num_transformer_layers, dropout, **kwargs)
        self.decoder = nn.Sequential(
            einops.layers.torch.Reduce("batch tokens features -> batch features", reduction="mean"),
            MLP(n_out_enc, n_hidden_dec, n_out_dec, activation)
        )
        
    def forward(self, X):
        z_enc = self.encoder(X)
        z = self.decoder(z_enc)
        return z

class PersTransformerSeq(torch.nn.Module):
    def __init__(self, n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, num_transformer_layers, 
                 n_hidden_dec=16, n_out_dec=5, dropout=.1, **kwargs):
        super(PersTransformerSeq, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, 
                               num_transformer_layers, dropout, **kwargs)
        self.decoder = nn.Sequential(
            MLP(n_out_enc, n_hidden_dec, n_out_dec, activation)
        )
        
    def forward(self, X):
        z_enc = self.encoder(X)
        z = self.decoder(z_enc)
        return z

class PersTransformerRegressor(torch.nn.Module):
    def __init__(self, n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, num_transformer_layers, 
                 pred_feats, pred_window, n_seq, 
                 dropout=.1, **kwargs):
        super(PersTransformerRegressor, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc, n_head_enc, dim_feed_enc, activation, 
                               num_transformer_layers, dropout, **kwargs)
        self.decoder = nn.Sequential(
                    ChannelMixerOut(n_seq, out_dim=pred_feats, dropout=dropout, ),
                    TokenMixerOut(n_out_enc, out_dim=pred_window, dropout=dropout, ),
                    Rearrange('b c t -> b t c')
        )
        
    def forward(self, X):
        z_enc = self.encoder(X)
        z = self.decoder(z_enc)
        return z