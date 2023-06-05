import torch
import torch.nn as nn
from torch import einsum
from collections import OrderedDict
from models.SpecAugment import SpecAugmentation

from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
from models.net_vlad import NetVLAD



def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwags):
        #out_w, out_h = 1, 1
        output = self.norm(x)
        output = self.fn(output, **kwags)
        ##output = self.fn(x, **kwags)
        return output


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):

        super(FeedForward, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim, hidden_dim)),
            ('ac1', nn.GELU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_dim, dim)),
            ('dropout2', nn.Dropout(dropout))
        ]))

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):

    def __init__(self, dim, heads=2, dim_head=128, dropout=0.):
    #def __init__(self, dim, heads=2, dropout=0.1):
        '''
        dim: dim of input
        dim_head: dim of q, k, v
        '''
        super(Attention, self).__init__()
        #dim_head = dim
        #dim_head = dim_head
        inner_dim = dim_head * heads
        project_out = not(heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, inner_dim * 3)

        #self.is_vlad = config.training.vlad
        #if self.is_vlad:
        #self.net_vlad = NetVLAD(cluster_size=20, feature_size=128)

        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # b:batch h:2 head number n: feature_dem d: time_dem

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), qkv)

        #q = q.transpose(2, 1)
        #k = k.transpose(2, 1)
        #v = v.transpose(2, 1)

        ##q = torch.squeeze(q, 1)
        ##k = torch.squeeze(k, 1)
        ##v = torch.squeeze(v, 1)          # (batch*2, time_dem, feature_dem)

        #qq = self.net_vlad(q)
        #kk = self.net_vlad(k)
        #vv = self.net_vlad(v)

        #q = qq.transpose(2, 1)
        #k = kk.transpose(2, 1)
        #v = vv.transpose(2, 1)

        #q = q.view(b, h, n, q.shape[2])
        #k = k.view(b, h, n, k.shape[2])
        #v = v.view(b, h, n, v.shape[2])
        #q = (  )                     # (batch, h=2, feature_dem, time_dem)
        #k =
        #v =

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)    #question 1: reduce the deminsion of q, k, v the final output demision shi shenme
        # question2: attention shi zai time dimension or doing it at the embeding deimension? tp look at the paper abouit scale vision transformer
        return out                 #this is the final result


class Transformer(nn.Module):                #mlp

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        #self.dim_head = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AudioTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super(AudioTransformer, self).__init__()
    #def __init__(self, x, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.):
        #patch_height, patch_width = pair(patch_size)
        #self.src = src
        #self.dim = dim
        #self.dim_head = dim
        #self.depth = depth
        #self.heads = heads
        #self.dropout = dropout
        #patch_dim = patch_height * patch_width  # 64 * 4 = 256 (16 * 16)

        #self.bn0 = nn.BatchNorm2d(64)

        #self.patch_embed = nn.Sequential(OrderedDict([
            #('rerange', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)),
            #('proj', nn.Linear(patch_dim, dim))
        #]))

        #self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                               #time_stripes_num=2,
                                               #freq_drop_width=8,
                                               #freq_stripes_num=2,
                                               #mask_type='zero_value')

        #self.pos_embedding = nn.Parameter(torch.randn(1, 125 + 1, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #self.dropout = nn.Dropout(dropout)

        self.blocks = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()     # consider maybe we need not use it

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )              #the second dim value is the number of classes


    def forward(self, x):

        #x = spec.unsqueeze(1)
        #x = x.transpose(1, 3)
        #x = self.bn0(x)
        #x = x.transpose(1, 3)
        #if self.training:
            #x = self.spec_augmenter(x)
        #x = self.patch_embed(x)
        b, n, _ = x.shape
        #b: batch_size  n:time dimension    _:embinding dimension

        #cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        #x = torch.cat((cls_token, x), dim=1)
        #x += self.pos_embedding[:, :(n + 1)]
        #x = self.dropout(x)       #randomly zeroes some of the elements of the input tensor

        #x = squeeze()
        x = self.blocks(x)        # here input of the sequenceVlad

        x = self.to_latent(x)            #output x
        xx = self.mlp_head(x)          ###need to rewrite

        return xx                      #delete mlp_head

if __name__ == '__main__':
    num_classes = 527
    #patch_size = (4, 64)
    embed_dim = 768      #768
    depth = 12
    num_heads = 12
    mlp_dim = 3072
    dropout = 0.1
    model = AudioTransformer(num_classes,
                             embed_dim,
                             depth,
                             num_heads,
                             mlp_dim,
                             dropout=dropout)
    feature = torch.randn(32, 500, 64)
    output = model(feature)
    print(output)