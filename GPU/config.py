"""Configuration dataclasses for the GPU training scripts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainerConfig:
    """Configuration options shared across all Lightning trainers."""

    max_epochs: int = 50
    min_epochs: int = 5
    accelerator: str = "auto"
    devices: Optional[int | list[int]] = None
    precision: str | int = "32-true"
    gradient_clip_val: Optional[float] = None
    deterministic: bool = True
    log_every_n_steps: int = 50
    accumulate_grad_batches: int = 1
    enable_checkpointing: bool = True
    fast_dev_run: bool | int = False


@dataclass
class WandbConfig:
    """Configuration options for Weights & Biases logging."""

    project: str = "ddi-fp-graph"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_model: bool = True
    save_dir: Optional[str] = None


@dataclass
class OptimizerConfig:
    """Configuration for model optimizers."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class GraphModelConfig:
    """Configuration for :class:`~GPU.models.GraphModel`."""

    gnn_name: str = "GATConv"
    gnn_nlayers: int = 4
    gnn_in: int = 9
    gnn_hid: int = 256
    dec_nlayers: int = 4
    dec_hid: int = 256
    attn_heads: int = 4
    dropout: float = 0.5
    act: str = "leakyrelu"
    final_concat: bool = True
    top_k: int = 5


@dataclass
class FingerprintMLPConfig:
    """Configuration for :class:`~GPU.models.FPMLP`."""

    enc_layers: int = 4
    in_dim: int = 2048
    hid_dim: int = 256
    dropout: float = 0.2
    act: str = "leakyrelu"
    fusion: str = "fingerprint_symmetric"
    batch_norm: bool = False
    top_k: int = 5
    concat: str | None = None

    def __post_init__(self) -> None:
        if self.concat is not None and self.fusion == "fingerprint_symmetric":
            # Legacy configs may only specify `concat`; mirror that selection in fusion.
            self.fusion = self.concat


@dataclass
class FingerprintGBDTConfig:
    """Shared configuration for gradient boosting fingerprint models."""

    fusion: str = "fingerprint_symmetric"
    top_k: int = 5
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


# Backwards compatible alias for legacy imports
FingerprintModelConfig = FingerprintMLPConfig


@dataclass
class FingerprintGraphModelConfig(GraphModelConfig):
    """Extension of :class:`GraphModelConfig` including fingerprint encoder options."""

    fp_nlayers: int = 4
    fp_in: int = 2048
    fp_hid: int = 256


@dataclass
class SSIDDIModelConfig:
    """Configuration for :class:`~GPU.models.SSIDDIModel`."""

    act: str = "leakyrelu"
    in_dim: int = 9
    hid_dim: int = 256
    GAT_head_dim: int = 64
    GAT_nheads: int = 4
    GAT_nlayers: int = 5
    top_k: int = 5


@dataclass
class DataModuleConfig:
    """Configuration that applies to all Lightning DataModules."""

    data_dir: str = "Data"
    batch_size: int = 256
    num_workers: int = 8
    kind: str = "morgan"
    include_neg: bool = True
    mode: str = "transductive"
    train_prop: float = 0.8
    val_prop: float = 0.5
    radius: int = 2
    nbits: int = 2048


@dataclass
class ExperimentConfig:
    """Top-level configuration used by :mod:`GPU.train`."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    seed: int = 2023
    model_name: str = "graph"

