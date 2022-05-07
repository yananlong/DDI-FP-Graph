import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics import AUROC, Accuracy, FBetaScore


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


class _FPModel(LightningModule):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        top_k: int = 5,
    ):
        super().__init__()

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=out_dim)
        self.topkacc = Accuracy(num_classes=out_dim, top_k=top_k)
        self.f1_macro = FBetaScore(num_classes=out_dim, beta=1.0, average="macro")
        self.f1_weighted = FBetaScore(num_classes=out_dim, beta=1.0, average="weighted")
        self.auroc = AUROC(num_classes=out_dim, average="weighted")

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        self.dec = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, drug1, drug2):
        d1 = self.enc(drug1)
        d2 = self.enc(drug2)
        d = d1 + d2
        d = self.dec(d)
        return d

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-3, weight_decay=0)

        return optimizer

    def training_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)

        # Logging
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_f1_macro", f1_m, on_step=False, on_epoch=True)
        self.log("train_f1_weighted", f1_w, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return {"accuracy": acc, "f1 macro": f1_m, "f1_weighted": f1_w, "loss": loss}

    def validation_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)

        # Logging
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", f1_m, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1_weighted", f1_w, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, loss

    def test_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)
        ypreds_probs = ypreds.softmax(dim=-1)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        auroc = self.auroc(ypreds_probs, y)

        # Logging
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1_macro", f1_m, prog_bar=True)
        self.log("test_f1_weighted", f1_w, prog_bar=True)
        self.log("test_auroc", auroc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, auroc, loss


class FPModel(LightningModule):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        nlayers: int = 4,
        dropout: int = 0,
        act: str = "leakyrelu",
        batch_norm: bool = False,
        concat: str = "final",
        top_k: int = 5,
    ):
        super().__init__()
        self.concat = concat

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=out_dim)
        self.topkacc = Accuracy(num_classes=out_dim, top_k=top_k)
        self.f1_macro = FBetaScore(num_classes=out_dim, beta=1.0, average="macro")
        self.f1_weighted = FBetaScore(num_classes=out_dim, beta=1.0, average="weighted")
        self.auroc = AUROC(num_classes=out_dim, average="weighted")

        # MLP
        if self.concat == "first":
            in_dim = 2 * in_dim
        hid_dim_factor = 2 if self.concat == "last" else 1

        self.enc = pyg_nn.MLP(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            out_channels=hid_dim,
            num_layers=nlayers,
            dropout=dropout,
            act=act,
            batch_norm=batch_norm,
        )
        self.dec = pyg_nn.MLP(
            in_channels=hid_dim * hid_dim_factor,
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_layers=nlayers,
            dropout=dropout,
            act=act,
            batch_norm=batch_norm,
        )

    def forward(self, drug1, drug2):
        # Encode
        if self.concat == "first":
            d = torch.cat([drug1, drug2], axis=1)
            d = d.type_as(d)
            d = self.enc(d)
        else:
            d1 = self.enc(drug1)
            d2 = self.enc(drug2)
            d = torch.cat([d1, d2], axis=1) if self.concat == "last" else d1 + d2
            d = d.type_as(d)

        # Decode
        d = self.dec(d)

        return d

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-3, weight_decay=0)

        return optimizer

    def training_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)

        # Logging
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_f1_macro", f1_m, on_step=False, on_epoch=True)
        self.log("train_f1_weighted", f1_w, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return {"accuracy": acc, "f1 macro": f1_m, "f1_weighted": f1_w, "loss": loss}

    def validation_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)

        # Logging
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", f1_m, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1_weighted", f1_w, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, loss

    def test_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)
        ypreds_probs = ypreds.softmax(dim=-1)

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        auroc = self.auroc(ypreds_probs, y)

        # Logging
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1_macro", f1_m, prog_bar=True)
        self.log("test_f1_weighted", f1_w, prog_bar=True)
        self.log("test_auroc", auroc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, auroc, loss


class GraphModel(LightningModule):
    def __init__(
        self,
        batch_size,
        act,
        gnn_name,
        gnn_nlayers,
        gnn_in,
        gnn_hid,
        dec_nlayers,
        dec_hid,
        out_dim,
        final_concat=False,
        dropout=0.5,
        top_k=5,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.gnn_layer = gnn_name
        self.layer_name = gnn_name.__name__
        self.final_concat = final_concat
        self.dec_in_fac = 2 if self.final_concat else 1
        self.dropout = dropout

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=out_dim)
        self.topkacc = Accuracy(num_classes=out_dim, top_k=top_k)
        self.f1_macro = FBetaScore(num_classes=out_dim, beta=1.0, average="macro")
        self.f1_weighted = FBetaScore(num_classes=out_dim, beta=1.0, average="weighted")
        self.auroc = AUROC(num_classes=out_dim, average="weighted")

        # Layers
        # Fingerprint encoder and global decoder
        self.dec = pyg_nn.MLP(
            in_channels=self.dec_in_fac * gnn_nlayers * gnn_hid,
            hidden_channels=dec_hid,
            out_channels=out_dim,
            num_layers=dec_nlayers,
            dropout=dropout,
            act=act,
        )

        # Atom encoder
        self.atom_enc = nn.Linear(gnn_in, gnn_hid)

        # GNN
        print("GNN used:", self.layer_name, sep=" ", flush=True)
        if self.layer_name in ["GINConv", "GINEConv"]:
            self.aux_nn = nn.Sequential(
                pyg_nn.MLP(
                    in_channels=gnn_hid,
                    hidden_channels=gnn_hid,
                    out_channels=gnn_hid,
                    num_layers=4,
                    dropout=dropout,
                    act=act,
                )
            )
            self.gnn_layers = nn.ModuleList(
                [self.gnn_layer(nn=self.aux_nn, edge_dim=3) for _ in range(gnn_nlayers)]
            )
        else:
            self.gnn_layers = nn.ModuleList(
                [self.gnn_layer(gnn_hid, gnn_hid) for _ in range(gnn_nlayers)]
            )

    def forward(self, batch):
        # Unpack batch
        (x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2,) = (
            batch.x1,
            batch.edge_index1,
            batch.edge_attr1,
            batch.x1_batch,
            batch.x2,
            batch.edge_index2,
            batch.edge_attr2,
            batch.x2_batch,
        )

        # Atom/node features
        x1 = self.atom_enc(x1.float())
        x2 = self.atom_enc(x2.float())

        # GNN operations
        graph_embeds1 = []
        graph_embeds2 = []
        for layer in self.gnn_layers:
            # Use edge attributes
            if self.layer_name in ["GATConv", "GATv2Conv", "GINEConv"]:
                x1 = F.relu(layer(x1, edge_index1, edge_attr1.float()))
                x2 = F.relu(layer(x2, edge_index2, edge_attr2.float()))
            # TODO: Use edge weights
            elif self.layer_name in ["GraphConv", "GatedGraphConv"]:
                raise NotImplementedError("Using edge weights is currently unsupported")
            else:
                x1 = F.relu(layer(x1, edge_index1))
                x2 = F.relu(layer(x2, edge_index2))

            # Collect graph embeddings
            graph_embeds1.append(pyg_nn.global_mean_pool(x1, batch1))
            graph_embeds2.append(pyg_nn.global_mean_pool(x2, batch2))

        # Aggregate
        g1 = torch.cat(graph_embeds1, axis=1)
        g2 = torch.cat(graph_embeds2, axis=1)

        if self.final_concat:
            g = torch.cat([g1, g2], axis=1)
        else:
            g = g1 + g2

        # Output
        out = self.dec(g)

        return out

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-3, weight_decay=0)

        return optimizer

    def training_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        # auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, batch_size=self.batch_size
        )
        self.log(
            "train_f1_macro",
            f1_m,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_f1_weighted",
            f1_w,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log("train_loss", loss)

        return {"accuracy": acc, "f1 macro": f1_m, "f1_weighted": f1_w, "loss": loss}

    def validation_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        # auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_f1_macro",
            f1_m,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_f1_weighted",
            f1_w,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("val_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_f1_macro",
            f1_m,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_f1_weighted",
            f1_w,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return acc, f1_m, f1_w, auroc, loss


class FPGraphModel(LightningModule):
    def __init__(
        self,
        batch_size,
        act,
        fp_nlayers,
        fp_in,
        fp_hid,
        gnn_name,
        gnn_nlayers,
        gnn_in,
        gnn_hid,
        dec_nlayers,
        dec_hid,
        out_dim,
        final_concat=False,
        dropout=0.5,
        top_k=5,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.gnn_layer = gnn_name
        self.layer_name = gnn_name.__name__
        self.final_concat = final_concat
        self.dec_in_fac = 2 if self.final_concat else 1
        self.dropout = dropout

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=out_dim)
        self.topkacc = Accuracy(num_classes=out_dim, top_k=top_k)
        self.f1_macro = FBetaScore(num_classes=out_dim, beta=1.0, average="macro")
        self.f1_weighted = FBetaScore(num_classes=out_dim, beta=1.0, average="weighted")
        self.auroc = AUROC(num_classes=out_dim, average="weighted")

        # Layers
        # Fingerprint encoder and global decoder
        self.fp_enc = pyg_nn.MLP(
            in_channels=fp_in,
            hidden_channels=fp_hid,
            out_channels=fp_hid,
            num_layers=fp_nlayers,
            dropout=dropout,
            act=act,
        )
        self.dec = pyg_nn.MLP(
            in_channels=self.dec_in_fac * (fp_hid + gnn_nlayers * gnn_hid),
            hidden_channels=dec_hid,
            out_channels=out_dim,
            num_layers=dec_nlayers,
            dropout=dropout,
            act=act,
        )

        # Atom encoder
        self.atom_enc = nn.Linear(gnn_in, gnn_hid)

        # GNN
        print("GNN used:", self.layer_name, sep=" ", flush=True)
        if self.layer_name in ["GINConv", "GINEConv"]:
            self.aux_nn = nn.Sequential(
                pyg_nn.MLP(
                    in_channels=gnn_hid,
                    hidden_channels=gnn_hid,
                    out_channels=gnn_hid,
                    num_layers=4,
                    dropout=dropout,
                    act=act,
                )
            )
            self.gnn_layers = nn.ModuleList(
                [self.gnn_layer(nn=self.aux_nn, edge_dim=3) for _ in range(gnn_nlayers)]
            )
        else:
            self.gnn_layers = nn.ModuleList(
                [self.gnn_layer(gnn_hid, gnn_hid) for _ in range(gnn_nlayers)]
            )

    def forward(self, batch):
        # Unpack batch
        (
            fp1,
            x1,
            edge_index1,
            edge_attr1,
            batch1,
            fp2,
            x2,
            edge_index2,
            edge_attr2,
            batch2,
        ) = (
            batch.fp1,
            batch.x1,
            batch.edge_index1,
            batch.edge_attr1,
            batch.x1_batch,
            batch.fp2,
            batch.x2,
            batch.edge_index2,
            batch.edge_attr2,
            batch.x2_batch,
        )

        # Fingerprint
        fp_embed1 = self.fp_enc(fp1)
        fp_embed2 = self.fp_enc(fp2)

        # Atom/node features
        x1 = self.atom_enc(x1.float())
        x2 = self.atom_enc(x2.float())

        # GNN operations
        graph_embeds1 = []
        graph_embeds2 = []
        for layer in self.gnn_layers:
            # Use edge attributes
            if self.layer_name in ["GATConv", "GATv2Conv", "GINEConv"]:
                x1 = F.relu(layer(x1, edge_index1, edge_attr1.float()))
                x2 = F.relu(layer(x2, edge_index2, edge_attr2.float()))
            # TODO: Use edge weights
            elif self.layer_name in ["GraphConv", "GatedGraphConv"]:
                raise NotImplementedError("Using edge weights is currently unsupported")
            else:
                x1 = F.relu(layer(x1, edge_index1))
                x2 = F.relu(layer(x2, edge_index2))

            # Collect graph embeddings
            graph_embeds1.append(pyg_nn.global_mean_pool(x1, batch1))
            graph_embeds2.append(pyg_nn.global_mean_pool(x2, batch2))

        # Aggregate
        # Batch x (fp hidden + nlayers * graph hidden)
        g1 = torch.cat([fp_embed1] + graph_embeds1, axis=1)
        g2 = torch.cat([fp_embed2] + graph_embeds2, axis=1)

        if self.final_concat:
            g = torch.cat([g1, g2], axis=1)
        else:
            g = g1 + g2

        # Output
        out = self.dec(g)

        return out

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=1e-3, weight_decay=0)

        return optimizer

    def training_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        # auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, batch_size=self.batch_size
        )
        self.log(
            "train_f1_macro",
            f1_m,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_f1_weighted",
            f1_w,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log("train_loss", loss)

        return {"accuracy": acc, "f1 macro": f1_m, "f1_weighted": f1_w, "loss": loss}

    def validation_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        # auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "val_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_f1_macro",
            f1_m,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_f1_weighted",
            f1_w,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("val_loss", loss, prog_bar=True)

        return acc, f1_m, f1_w, loss

    def test_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.accuracy(ypreds, y)
        f1_m = self.f1_macro(ypreds, y)
        f1_w = self.f1_weighted(ypreds, y)
        auroc = self.auroc(ypreds, y)

        # Logging
        self.log(
            "test_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_f1_macro",
            f1_m,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_f1_weighted",
            f1_w,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return acc, f1_m, f1_w, auroc, loss
