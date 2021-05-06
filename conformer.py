
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from espnet.transformer.subsampling import Conv2dSubsampling_no_pos, Conv2dSubsampling, Conv2dSubsampling6, VGG2L_C, ResnetSubsampling
from espnet.nets_utils import get_activation, make_pad_mask
from espnet.transformer.embedding import RelPositionalEncoding

import pdb

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super(DepthWiseConv1d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super(Scale, self).__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        temp = self.to_kv(context).chunk(2, dim = -1) # make python 2 compatible
        q, k, v = (self.to_q(x), temp[0], temp[1])

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super(ConformerConvModule, self).__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x

# Conformer Encoder

class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        config
    ):
        super(ConformerEncoder, self).__init__()

        if config.subsampling == 'conv2dnopos':
            Subsampling_layer = Conv2dSubsampling_no_pos
        elif config.subsampling == 'conv2d':
            Subsampling_layer = Conv2dSubsampling
        elif config.subsampling == 'conv2d6':
            Subsampling_layer = Conv2dSubsampling6
        elif config.subsampling == 'vgg2l':
            Subsampling_layer = VGG2L_C
        elif config.subsampling == 'resnet':
            Subsampling_layer = ResnetSubsampling
        else:
            raise NotImplementedError

        self.subsampling = Subsampling_layer(
            input_size,
            config.dim,
            config.conv_dropout,
            # RelPositionalEncoding(
            #     config.dim,
            #     config.positional_dropout_rate
            # )
        )

        self.feedforward = nn.Linear(
            config.dim, config.dim
        )
        self.dropout = nn.Dropout(config.ff_dropout)

        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=config.dim,
                dim_head=config.dim_head,
                heads=config.heads,
                ff_mult=config.ff_mult,
                conv_expansion_factor=config.conv_expansion_factor,
                conv_kernel_size=config.conv_kernel_size,
                attn_dropout=config.attn_dropout,
                ff_dropout=config.ff_dropout,
                conv_dropout=config.conv_dropout
            ) for _ in range(config.layers)
        ])

    def forward(
        self,
        xs_pad,
        ilens,
        prev_states = None
    ):
        # xs_pad.shape = [4, 1556, 81]
        if isinstance(self.subsampling, ResnetSubsampling):
            xs_pad, masks = self.subsampling(xs_pad, ilens)
            xs_pad = self.dropout(xs_pad)

            # xs_pad = [4, 519, 256]
            # masks  = [4, 519], False = padding

        else:
            masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
            xs_pad, masks = self.subsampling(xs_pad, masks)

            masks = masks.squeeze(1)
            xs_pad = self.dropout(self.feedforward(xs_pad))

            # xs_pad = [4, 259, 256]
            # masks  = [4, 259], False = padding


        for layer in self.layers:
            xs_pad = layer(xs_pad, masks)

        olens = masks.sum(1)

        return xs_pad, olens, None

    def add_noise(self):
        with torch.no_grad():
            for name, param in self.layers.named_parameters():
                if 'weight' in name:
                    param.add_(torch.randn(param.size()).to(param.device) * 0.05)