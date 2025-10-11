"""Configuration dataclasses for the PyTorch training scripts."""
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
    """Configuration for :class:`~PyTorch.models.GraphModel`."""

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
class FingerprintModelConfig:
    """Configuration for :class:`~PyTorch.models.FPModel`."""

    enc_layers: int = 4
    in_dim: int = 2048
    hid_dim: int = 256
    dropout: float = 0.2
    act: str = "leakyrelu"
    concat: str = "last"
    batch_norm: bool = False
    top_k: int = 5


@dataclass
class FingerprintGraphModelConfig(GraphModelConfig):
    """Extension of :class:`GraphModelConfig` including fingerprint encoder options."""

    fp_nlayers: int = 4
    fp_in: int = 2048
    fp_hid: int = 256


@dataclass
class SSIDDIModelConfig:
    """Configuration for :class:`~PyTorch.models.SSIDDIModel`."""

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
    """Top-level configuration used by :mod:`PyTorch.train`."""

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    seed: int = 2023
    model_name: str = "graph"

