from __future__ import annotations

from typing import Any, Callable, Dict, Optional

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

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .ssiddi import SSI_DDI


EPS = 1e-12


class FPMLP(pl.LightningModule):
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


class BaseFingerprintGBDT(pl.LightningModule):
    """Common utilities for gradient boosting models that operate on FPs."""

    def __init__(
        self,
        out_dim: int,
        fusion: str = "fingerprint_symmetric",
        top_k: int = 5,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["estimator_kwargs"])

        alias_map = {
            "first": "fingerprint_concat",
            "concat": "fingerprint_concat",
        }
        fusion_mode = alias_map.get(fusion.lower(), fusion.lower())
        valid_fusions = {"fingerprint_concat", "fingerprint_symmetric"}
        if fusion_mode not in valid_fusions:
            raise ValueError(
                "Unsupported fusion mode for gradient boosting models. "
                f"Expected one of {sorted(valid_fusions)}, got '{fusion}'."
            )

        self.fusion = fusion_mode
        self.out_dim = out_dim
        self.top_k = top_k
        self.estimator_kwargs: Dict[str, Any] = dict(estimator_kwargs or {})
        self.loss_module = nn.NLLLoss()
        self.automatic_optimization = False

        metric_kwargs = {"task": "multiclass", "num_classes": out_dim}
        self.accuracy = Accuracy(**metric_kwargs)
        self.f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.auroc = AUROC(average="macro", **metric_kwargs)

        self.test_accuracy = Accuracy(**metric_kwargs)
        self.test_f1_macro = FBetaScore(beta=1.0, average="macro", **metric_kwargs)
        self.test_f1_weighted = FBetaScore(beta=1.0, average="weighted", **metric_kwargs)
        self.test_auroc = AUROC(average="macro", **metric_kwargs)

        self._train_features: list[np.ndarray] = []
        self._train_labels: list[torch.Tensor] = []
        self._estimator = self._build_estimator(out_dim)
        self._is_fitted = False

    # ---------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    def _build_estimator(self, out_dim: int):  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _fuse_inputs(self, drug1: torch.Tensor, drug2: torch.Tensor) -> torch.Tensor:
        if self.fusion == "fingerprint_concat":
            return torch.cat([drug1, drug2], dim=1)
        if self.fusion == "fingerprint_symmetric":
            union = drug1 + drug2
            intersection = drug1 * drug2
            exclusive = torch.abs(drug1 - drug2)
            return torch.cat([union, intersection, exclusive], dim=1)
        raise RuntimeError(f"Unsupported fusion mode '{self.fusion}'.")  # pragma: no cover

    def _prepare_features(self, drug1: torch.Tensor, drug2: torch.Tensor) -> np.ndarray:
        fused = self._fuse_inputs(drug1.detach().cpu(), drug2.detach().cpu())
        return fused.numpy().astype(np.float32, copy=False)

    def _collect_batch(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        drug1, drug2, labels = batch
        self._train_features.append(self._prepare_features(drug1, drug2))
        self._train_labels.append(labels.detach().cpu())

    def _fit_estimator(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._estimator.fit(features, labels)

    def _predict_proba_numpy(self, drug1: torch.Tensor, drug2: torch.Tensor) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "The gradient boosting estimator has not been fitted yet. "
                "Call `trainer.fit` before requesting predictions."
            )
        features = self._prepare_features(drug1, drug2)
        probs = self._estimator.predict_proba(features)
        return np.asarray(probs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # pragma: no cover - data collection
        self._collect_batch(batch)
        return None

    def on_train_epoch_start(self) -> None:  # pragma: no cover - simple state reset
        self._train_features.clear()
        self._train_labels.clear()

    def on_train_epoch_end(self) -> None:
        if not self._train_features:
            return
        features = np.concatenate(self._train_features, axis=0)
        labels = torch.cat(self._train_labels).cpu().numpy()
        self._fit_estimator(features, labels)
        self._is_fitted = True

        probs = torch.from_numpy(self._estimator.predict_proba(features)).float()
        targets = torch.from_numpy(labels).long()
        log_probs = torch.log(probs.clamp_min(EPS))
        loss = self.loss_module(log_probs, targets)
        accuracy = (probs.argmax(dim=1) == targets).float().mean()

        batch_size = features.shape[0]
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

    def forward(self, drug1: torch.Tensor, drug2: torch.Tensor) -> torch.Tensor:
        probs = self._predict_proba_numpy(drug1, drug2)
        return torch.from_numpy(probs).to(drug1.device, dtype=torch.float32)

    def _shared_eval_step(self, batch, stage: str) -> torch.Tensor:
        drug1, drug2, labels = batch
        probs = self(drug1, drug2)
        log_probs = torch.log(probs.clamp_min(EPS))
        loss = self.loss_module(log_probs, labels)

        if stage == "val":
            acc = self.accuracy(probs, labels)
            f1_m = self.f1_macro(probs, labels)
            f1_w = self.f1_weighted(probs, labels)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
            self.log("val_f1_macro", f1_m, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
            self.log(
                "val_f1_weighted",
                f1_w,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.size(0),
            )
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        elif stage == "test":
            self.test_accuracy(probs, labels)
            self.test_f1_macro(probs, labels)
            self.test_f1_weighted(probs, labels)
            self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
            self.log(
                "test_f1_macro",
                self.test_f1_macro,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.size(0),
            )
            self.log(
                "test_f1_weighted",
                self.test_f1_weighted,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=labels.size(0),
            )
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        drug1, drug2, labels = batch
        probs = self(drug1, drug2)
        return {"y": labels, "ypreds": probs}

    def configure_optimizers(self):  # pragma: no cover - no optimizers for GBDT
        return None


class FPCatBoostModel(BaseFingerprintGBDT):
    def __init__(
        self,
        out_dim: int,
        fusion: str = "fingerprint_symmetric",
        top_k: int = 5,
        depth: int = 8,
        learning_rate: float = 0.1,
        iterations: int = 1000,
        l2_leaf_reg: float = 3.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        params: Dict[str, Any] = dict(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
        )
        if random_state is not None:
            params["random_seed"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
        )

    def _build_estimator(self, out_dim: int) -> CatBoostClassifier:
        params = dict(self.estimator_kwargs)
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("classes_count", out_dim)
        return CatBoostClassifier(**params)


class FPLightGBMModel(BaseFingerprintGBDT):
    def __init__(
        self,
        out_dim: int,
        fusion: str = "fingerprint_symmetric",
        top_k: int = 5,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        n_estimators: int = 500,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        params: Dict[str, Any] = dict(
            objective="multiclass",
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=-1,
        )
        if random_state is not None:
            params["random_state"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
        )

    def _build_estimator(self, out_dim: int) -> LGBMClassifier:
        params = dict(self.estimator_kwargs)
        params.setdefault("objective", "multiclass")
        params.setdefault("num_class", out_dim)
        return LGBMClassifier(**params)


class FPXGBoostModel(BaseFingerprintGBDT):
    def __init__(
        self,
        out_dim: int,
        fusion: str = "fingerprint_symmetric",
        top_k: int = 5,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 500,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        params: Dict[str, Any] = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            gamma=gamma,
            objective="multi:softprob",
            use_label_encoder=False,
            tree_method="hist",
            eval_metric="mlogloss",
        )
        if random_state is not None:
            params["random_state"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
        )

    def _build_estimator(self, out_dim: int) -> XGBClassifier:
        params = dict(self.estimator_kwargs)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", out_dim)
        params.setdefault("use_label_encoder", False)
        return XGBClassifier(**params)


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


# Backwards compatible aliases for legacy imports
FP_MLP = FPMLP
FPModel = FPMLP
FP_CatBoost = FPCatBoostModel
FPCatBoost = FPCatBoostModel
FP_LightGBM = FPLightGBMModel
FPLightGBM = FPLightGBMModel
FP_XGBoost = FPXGBoostModel
FPXGBoost = FPXGBoostModel
