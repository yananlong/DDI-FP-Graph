"""Lightning-compatible wrappers for gradient boosting fingerprint models."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import pytorch as pl
from torch import nn
from torchmetrics.classification import Accuracy, AUROC, FBetaScore

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


EPS = 1e-12


def _normalize_device(device: Optional[str]) -> str:
    """Resolve the execution device, defaulting to CUDA when available."""

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    device_lower = device.lower()
    if device_lower in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn("CUDA requested but not available; falling back to CPU.", stacklevel=2)
        return "cpu"
    if device_lower == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device selection '{device}'. Use 'cpu' or 'cuda'.")


class BaseFingerprintGBDT(pl.LightningModule):
    """Common utilities for gradient boosting models that operate on FPs."""

    def __init__(
        self,
        out_dim: int,
        fusion: str = "fingerprint_symmetric",
        top_k: int = 5,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["estimator_kwargs"])

        fusion_mode = (fusion or "").lower()
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
        self.device_kind = _normalize_device(device)

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

    # ------------------------------------------------------------------
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
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        device_kind = _normalize_device(device)
        params: Dict[str, Any] = dict(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
        )
        if random_state is not None:
            params["random_seed"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        params.setdefault("task_type", "GPU" if device_kind == "cuda" else "CPU")
        if device_kind == "cuda":
            params.setdefault("devices", "0")

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
            device=device_kind,
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
        min_child_samples: int = 20,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        device_kind = _normalize_device(device)
        params: Dict[str, Any] = dict(
            objective="multiclass",
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=-1,
        )
        if random_state is not None:
            params["random_state"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        params.setdefault("device_type", "gpu" if device_kind == "cuda" else "cpu")

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
            device=device_kind,
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
        min_child_weight: float = 1.0,
        reg_alpha: float = 0.0,
        random_state: Optional[int] = None,
        estimator_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        device_kind = _normalize_device(device)
        params: Dict[str, Any] = dict(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            objective="multi:softprob",
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        if random_state is not None:
            params["random_state"] = random_state
        if estimator_kwargs:
            params.update(estimator_kwargs)

        params.setdefault("tree_method", "gpu_hist" if device_kind == "cuda" else "hist")
        if device_kind == "cuda":
            params.setdefault("predictor", "gpu_predictor")

        super().__init__(
            out_dim=out_dim,
            fusion=fusion,
            top_k=top_k,
            estimator_kwargs=params,
            device=device_kind,
        )

    def _build_estimator(self, out_dim: int) -> XGBClassifier:
        params = dict(self.estimator_kwargs)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", out_dim)
        params.setdefault("use_label_encoder", False)
        return XGBClassifier(**params)


__all__ = [
    "BaseFingerprintGBDT",
    "FPCatBoostModel",
    "FPLightGBMModel",
    "FPXGBoostModel",
]
