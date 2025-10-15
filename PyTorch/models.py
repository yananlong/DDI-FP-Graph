from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from lightning import pytorch as pl
from torchmetrics.classification import (
    Accuracy,
    AUROC,
    FBetaScore,
)

from .ssiddi import SSI_DDI


class FPModel(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        nlayers: int = 4,
        dropout: float = 0.0,
        act: str = "leakyrelu",
        batch_norm: bool = False,
        fusion: str = "fingerprint_symmetric",
        concat: str | None = None,
        top_k: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["top_k"])

        fusion_mode = fusion or ""
        alias_map = {
            "first": "fingerprint_concat",
            "last": "embedding_concat",
            "final": "embedding_sum",
            "sum": "embedding_sum",
        }
        if concat:
            # Allow legacy configs that still specify `concat` to override the fusion mode.
            fusion_mode = alias_map.get(concat.lower(), concat.lower())
        fusion_mode = alias_map.get(fusion_mode.lower(), fusion_mode.lower())

        valid_fusions = {
            "fingerprint_concat",
            "fingerprint_symmetric",
            "embedding_concat",
            "embedding_sum",
            "embedding_symmetric",
        }
        if fusion_mode not in valid_fusions:
            raise ValueError(
                "Unsupported fusion mode. Expected one of "
                f"{sorted(valid_fusions)}, got '{fusion_mode}'."
            )

        self.fusion = fusion_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        metric_kwargs = {"task": "multiclass", "num_classes": out_dim}
        self.accuracy = Accuracy(**metric_kwargs)
        self.f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.auroc = AUROC(average="macro", **metric_kwargs)

        self.test_accuracy = Accuracy(**metric_kwargs)
        self.test_f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.test_f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.test_auroc = AUROC(average="macro", **metric_kwargs)

        # MLP
        if self.fusion == "fingerprint_concat":
            enc_in_dim = 2 * in_dim
        elif self.fusion == "fingerprint_symmetric":
            enc_in_dim = 3 * in_dim
        else:
            enc_in_dim = in_dim

        if self.fusion in {"embedding_concat", "embedding_symmetric"}:
            dec_in_dim = 2 * hid_dim
        else:
            dec_in_dim = hid_dim

        self.enc = pyg_nn.MLP(
            in_channels=enc_in_dim,
            hidden_channels=hid_dim,
            out_channels=hid_dim,
            num_layers=nlayers,
            dropout=dropout,
            act=act,
            batch_norm=batch_norm,
        )
        self.dec = pyg_nn.MLP(
            in_channels=dec_in_dim,
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_layers=nlayers,
            dropout=dropout,
            act=act,
            batch_norm=batch_norm,
        )

    def forward(self, drug1, drug2):
        # Encode
        if self.fusion == "fingerprint_concat":
            fused = torch.cat([drug1, drug2], dim=1)
            d = self.enc(fused)
        elif self.fusion == "fingerprint_symmetric":
            union = drug1 + drug2
            intersection = drug1 * drug2
            exclusive = torch.abs(drug1 - drug2)
            fused = torch.cat([union, intersection, exclusive], dim=1)
            d = self.enc(fused)
        else:
            d1 = self.enc(drug1)
            d2 = self.enc(drug2)
            if self.fusion == "embedding_concat":
                d = torch.cat([d1, d2], dim=1)
            elif self.fusion == "embedding_sum":
                d = d1 + d2
            elif self.fusion == "embedding_symmetric":
                d = torch.cat([torch.abs(d1 - d2), d1 * d2], dim=1)
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"Unsupported fusion mode '{self.fusion}'.")

        # Decode
        d = self.dec(d)

        return d

    def configure_optimizers(self):
        optimizer_class = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }.get(self.optimizer_name.lower())

        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        optimizer = optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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

        return {"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "loss": loss}

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
        # Metrics
        loss = self.loss_module(ypreds, y)
        acc = self.test_accuracy(ypreds, y)
        f1_m = self.test_f1_macro(ypreds, y)
        f1_w = self.test_f1_weighted(ypreds, y)
        auroc = self.test_auroc(ypreds, y)

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1_macro", f1_m, prog_bar=True)
        self.log("test_f1_weighted", f1_w, prog_bar=True)
        self.log("test_auroc", auroc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        d1, d2, y = batch
        ypreds = self(d1, d2)

        return {"y": y, "ypreds": ypreds}


class GraphModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        act: str,
        gnn_name: Callable,
        gnn_nlayers: int,
        gnn_in: int,
        gnn_hid: int,
        dec_nlayers: int,
        dec_hid: int,
        attn_heads: int,
        out_dim: int,
        final_concat: bool = False,
        dropout: int = 0.5,
        top_k: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gnn_name"])
        self.batch_size = batch_size
        self.gnn_layer = gnn_name
        self.layer_name = gnn_name.__name__
        self.final_concat = final_concat
        self.dec_in_fac = 2 if self.final_concat else 1
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        metric_kwargs = {"task": "multiclass", "num_classes": out_dim}
        self.accuracy = Accuracy(**metric_kwargs)
        self.topkacc = Accuracy(top_k=top_k, **metric_kwargs)
        self.f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.auroc = AUROC(average="macro", **metric_kwargs)

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
        elif self.layer_name in ["GATConv", "GATv2Conv"]:
            self.gnn_layers = nn.ModuleList(
                [
                    self.gnn_layer(
                        gnn_hid, int(gnn_hid / attn_heads), attn_heads, edge_dim=3
                    )
                    for _ in range(gnn_nlayers)
                ]
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
        optimizer_class = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }.get(self.optimizer_name.lower())

        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        optimizer = optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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

        return {"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "loss": loss}

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
        self.log("test_acc", acc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1_macro", f1_m, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1_weighted", f1_w, prog_bar=True, batch_size=self.batch_size)
        self.log("test_auroc", auroc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return acc, f1_m, f1_w, auroc, loss


class FPGraphModel(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        act: str,
        fp_nlayers: int,
        fp_in: int,
        fp_hid: int,
        gnn_name: Callable,
        gnn_nlayers: int,
        gnn_in: int,
        gnn_hid: int,
        dec_nlayers: int,
        dec_hid: int,
        out_dim: int,
        final_concat: bool = False,
        dropout: int = 0.5,
        top_k: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gnn_name"])
        self.batch_size = batch_size
        self.gnn_layer = gnn_name
        self.layer_name = gnn_name.__name__
        self.final_concat = final_concat
        self.dec_in_fac = 2 if self.final_concat else 1
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        metric_kwargs = {"task": "multiclass", "num_classes": out_dim}
        self.accuracy = Accuracy(**metric_kwargs)
        self.topkacc = Accuracy(top_k=top_k, **metric_kwargs)
        self.f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.auroc = AUROC(average="macro", **metric_kwargs)

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
        elif self.layer_name in ["GATConv", "GATv2Conv"]:
            self.gnn_layers = nn.ModuleList(
                [
                    self.gnn_layer(
                        gnn_hid, int(gnn_hid / attn_heads), attn_heads, edge_dim=3
                    )
                    for _ in range(gnn_nlayers)
                ]
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
        optimizer_class = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }.get(self.optimizer_name.lower())

        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        optimizer = optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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

        return {"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "loss": loss}

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


class SSIDDIModel(pl.LightningModule):
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
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.GAT_head_dim = GAT_head_dim
        self.GAT_nheads = GAT_nheads
        self.GAT_nlayers = GAT_nlayers
        self.out_dim = out_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer

        # Prepare arguments
        self.att_dim = self.GAT_head_dim * self.GAT_nheads
        self.heads_out_feat_params = np.repeat(
            self.GAT_head_dim, self.GAT_nlayers
        ).tolist()
        self.blocks_params = np.repeat(self.GAT_nheads, self.GAT_nlayers).tolist()

        # Loss
        self.loss_module = nn.CrossEntropyLoss()

        metric_kwargs = {"task": "multiclass", "num_classes": out_dim}
        self.accuracy = Accuracy(**metric_kwargs)
        self.f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.auroc = AUROC(average="macro", **metric_kwargs)

        self.test_accuracy = Accuracy(**metric_kwargs)
        self.test_f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.test_f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.test_auroc = AUROC(average="macro", **metric_kwargs)

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
        optimizer_class = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }.get(self.optimizer_name.lower())

        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        optimizer = optimizer_class(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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

        return {"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "loss": loss}

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

        loss = self.loss_module(ypreds, y)
        acc = self.test_accuracy(ypreds, y)
        f1_m = self.test_f1_macro(ypreds, y)
        f1_w = self.test_f1_weighted(ypreds, y)
        auroc = self.test_auroc(ypreds, y)

        self.log("test_acc", acc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1_macro", f1_m, prog_bar=True, batch_size=self.batch_size)
        self.log("test_f1_weighted", f1_w, prog_bar=True, batch_size=self.batch_size)
        self.log("test_auroc", auroc, prog_bar=True, batch_size=self.batch_size)
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def predict_step(self, batch, batch_idx):
        # Forward pass
        ypreds = self(batch)
        y = batch.y

        return {"y": y, "ypreds": ypreds}
