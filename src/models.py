import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ssiddi import SSI_DDI
from torch import nn
from torchmetrics import AUROC, Accuracy, FBetaScore


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
        # Global decoder
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


class SSIDDIModel(LightningModule):
    def __init__(
        self,
        batch_size,
        act,
        in_dim,
        hid_dim,
        GAT_head_dim,
        GAT_nheads,
        GAT_nlayers,
        out_dim,
        top_k=5,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.GAT_head_dim = GAT_head_dim
        self.GAT_nheads = GAT_nheads
        self.GAT_nlayers = GAT_nlayers
        self.out_dim = out_dim

        # Prepare arguments
        self.att_dim = self.GAT_head_dim * self.GAT_nheads
        self.heads_out_feat_params = np.repeat(
            self.GAT_head_dim, self.GAT_nlayers
        ).tolist()
        self.blocks_params = np.repeat(self.GAT_nheads, self.GAT_nlayers).tolist()

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = Accuracy(num_classes=out_dim)
        self.topkacc = Accuracy(num_classes=out_dim, top_k=top_k)
        self.f1_macro = FBetaScore(num_classes=out_dim, beta=1.0, average="macro")
        self.f1_weighted = FBetaScore(num_classes=out_dim, beta=1.0, average="weighted")
        self.auroc = AUROC(num_classes=out_dim, average="weighted")

        # SSI-DDI layer
        self.ssi_ddi = SSI_DDI(
            act,
            self.in_dim,
            self.hid_dim,
            self.att_dim,
            self.out_dim,
            self.heads_out_feat_params,
            self.blocks_params,
        )

    def forward(self, batch):
        return self.ssi_ddi(batch)

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
