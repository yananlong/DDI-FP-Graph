from itertools import product

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch import nn


class SSI_DDI(nn.Module):
    def __init__(
        self,
        act,
        in_dim,
        hid_dim,
        att_dim,
        out_dim,
        heads_out_feat_params,
        blocks_params,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.att_dim = att_dim
        self.out_dim = out_dim
        self.n_blocks = len(blocks_params)
        self.initial_norm = pyg_nn.LayerNorm(self.in_dim)

        # GAT
        self.blocks = []
        self.net_norms = nn.ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(
            zip(heads_out_feat_params, blocks_params)
        ):
            block = SSI_DDI_Block(n_heads, in_dim, head_out_feats)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            in_dim = head_out_feats * n_heads
            self.net_norms.append(pyg_nn.LayerNorm(in_dim))
        print("SSI-DDI specs:", self.blocks, flush=True)

        # Self-attention
        self.co_attention = CoAttentionLayer(self.att_dim)

        # Decoder
        self.dec = pyg_nn.MLP(
            in_channels=self.att_dim * self.n_blocks ** 2,
            hidden_channels=self.hid_dim,
            out_channels=self.out_dim,
            num_layers=4,
            act=act,
        )

    def forward(self, batch):
        batch.x1 = self.initial_norm(batch.x1.float(), batch.x1_batch)
        batch.x2 = self.initial_norm(batch.x2.float(), batch.x2_batch)

        # GAT layers
        repr1 = []
        repr2 = []
        for i, block in enumerate(self.blocks):
            x1, r1 = block(batch.x1, batch.edge_index1, batch.x1_batch)
            x2, r2 = block(batch.x2, batch.edge_index2, batch.x2_batch)

            repr1.append(r1)
            repr2.append(r2)

            batch.x1 = F.elu(self.net_norms[i](x1, batch.x1_batch))
            batch.x2 = F.elu(self.net_norms[i](x2, batch.x2_batch))

        # Self-attention
        # repr*: [batch * n_blocks * attn_dim]
        # atts: [batch * n_blocks * n_blocks]
        repr1 = torch.stack(repr1, dim=-2)
        repr2 = torch.stack(repr2, dim=-2)
        atts = self.co_attention(repr1, repr2)

        # Normalize
        # https://github.com/kanz76/SSI-DDI/blob/master/layers.py#L47-L48
        repr1 = F.normalize(repr1, dim=-1)
        repr2 = F.normalize(repr2, dim=-1)

        # Reweight representations
        rws = []
        for i, j in product(range(self.n_blocks), range(self.n_blocks)):
            a = torch.unsqueeze(atts[:, i, j], dim=-1)
            r = repr1.select(1, i) + repr2.select(1, j)
            rw = r * a
            rws.append(rw)
        rws = torch.cat(rws, dim=-1)

        # Decode
        out = self.dec(rws)

        return out


class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_dim, head_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_dim = in_dim
        self.out_features = head_out_feats
        self.conv = pyg_nn.GATv2Conv(in_dim, head_out_feats, n_heads)
        self.readout = pyg_nn.SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, x, edge_index, batch_index):
        x = self.conv(x, edge_index)
        att_x, _, _, att_batch, _, _ = self.readout(x, edge_index, batch=batch_index)
        global_graph_emb = pyg_nn.global_add_pool(att_x, att_batch)

        return x, global_graph_emb


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        attentions = torch.tanh(e_activations) @ self.a

        return attentions
