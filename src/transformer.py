import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        # d_k = d_k.type_as(q)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, emb_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, emb_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, lin_dim, dropout=0):
        super().__init__()

        # Attention
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # MLP
        self.lin = nn.Sequential(
            nn.Linear(input_dim, lin_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lin_dim, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        lin_out = self.lin(x)
        x = x + self.dropout(lin_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, input_dim, num_heads, lin_dim, dropout=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(input_dim, num_heads, lin_dim, dropout=0)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)

        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for block in self.blocks:
            _, attn_map = block.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = block(x)

        return attention_maps
