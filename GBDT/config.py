"""Configuration dataclasses for gradient boosting fingerprint models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FingerprintGBDTConfig:
    """Shared configuration for :mod:`GBDT.models` estimators."""

    fusion: str = "fingerprint_symmetric"
    top_k: int = 5
    device: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def build_params(self) -> Dict[str, Any]:
        return dict(self.extra_params)


@dataclass
class FPCatBoostConfig(FingerprintGBDTConfig):
    depth: int = 8
    learning_rate: float = 0.1
    iterations: int = 1000
    l2_leaf_reg: float = 3.0
    bagging_temperature: float = 1.0
    random_strength: float = 1.0
    random_state: Optional[int] = None

    def build_params(self) -> Dict[str, Any]:
        params = super().build_params()
        params.setdefault("depth", self.depth)
        params.setdefault("learning_rate", self.learning_rate)
        params.setdefault("iterations", self.iterations)
        params.setdefault("l2_leaf_reg", self.l2_leaf_reg)
        params.setdefault("bagging_temperature", self.bagging_temperature)
        params.setdefault("random_strength", self.random_strength)
        params.setdefault("loss_function", "MultiClass")
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        if self.random_state is not None:
            params.setdefault("random_seed", self.random_state)
        return params


@dataclass
class FPLightGBMConfig(FingerprintGBDTConfig):
    num_leaves: int = 31
    learning_rate: float = 0.1
    n_estimators: int = 500
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: Optional[int] = None

    def build_params(self) -> Dict[str, Any]:
        params = super().build_params()
        params.setdefault("objective", "multiclass")
        params.setdefault("num_leaves", self.num_leaves)
        params.setdefault("learning_rate", self.learning_rate)
        params.setdefault("n_estimators", self.n_estimators)
        params.setdefault("subsample", self.subsample)
        params.setdefault("colsample_bytree", self.colsample_bytree)
        params.setdefault("min_child_samples", self.min_child_samples)
        params.setdefault("reg_alpha", self.reg_alpha)
        params.setdefault("reg_lambda", self.reg_lambda)
        params.setdefault("n_jobs", -1)
        if self.random_state is not None:
            params.setdefault("random_state", self.random_state)
        return params


@dataclass
class FPXGBoostConfig(FingerprintGBDTConfig):
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 500
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    random_state: Optional[int] = None

    def build_params(self) -> Dict[str, Any]:
        params = super().build_params()
        params.setdefault("max_depth", self.max_depth)
        params.setdefault("learning_rate", self.learning_rate)
        params.setdefault("n_estimators", self.n_estimators)
        params.setdefault("subsample", self.subsample)
        params.setdefault("colsample_bytree", self.colsample_bytree)
        params.setdefault("reg_lambda", self.reg_lambda)
        params.setdefault("gamma", self.gamma)
        params.setdefault("min_child_weight", self.min_child_weight)
        params.setdefault("reg_alpha", self.reg_alpha)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("use_label_encoder", False)
        params.setdefault("tree_method", "hist")
        params.setdefault("eval_metric", "mlogloss")
        if self.random_state is not None:
            params.setdefault("random_state", self.random_state)
        return params
