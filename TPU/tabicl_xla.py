"""PyTorch/XLA wrapper for loading pretrained TabICL encoders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # pragma: no cover - exercised in integration
    from tabicl import TabICL
except ImportError as exc:  # pragma: no cover - user action required
    raise ImportError(
        "The `tabicl` package is required to load pretrained checkpoints. "
        "Install it from https://github.com/soda-inria/tabicl before using the TPU pipeline."
    ) from exc


@dataclass
class TabICLCheckpointConfig:
    """Configuration for loading a pretrained TabICL encoder."""

    checkpoint_path: Path
    trainable: bool = False
    normalise: bool = True
    dtype: torch.dtype = torch.float32


class TabICLEncoder(nn.Module):
    """Thin module that exposes TabICL's feature extractor for TPU workloads."""

    def __init__(self, config: TabICLCheckpointConfig) -> None:
        super().__init__()
        checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
        if "config" not in checkpoint or "state_dict" not in checkpoint:
            raise ValueError(
                "TabICL checkpoint must contain `config` and `state_dict` entries. "
                "Use the upstream training pipeline or Hugging Face releases."
            )

        self.model = TabICL(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        if config.dtype != torch.float32:
            self.model.to(dtype=config.dtype)
        if not config.trainable:
            self.model.eval()
            self.model.requires_grad_(False)
        self._trainable = bool(config.trainable)
        self._normalise = bool(config.normalise)
        self._dtype = config.dtype
        self.output_dim = self.model.row_interactor.num_cls * self.model.row_interactor.embed_dim

    def _encode_flat(self, inputs: Tensor) -> Tensor:
        """Encode a batch of fingerprints of shape ``(B, F)`` into TabICL embeddings."""

        inputs = inputs.to(self._dtype, copy=False)
        with torch.set_grad_enabled(self._trainable):
            tables = inputs.unsqueeze(1)  # (B, F) -> (B, 1, F)
            features = self.model.col_embedder._train_forward(tables)
            rows = self.model.row_interactor._train_forward(features)
            reps = rows[:, 0, :]
        if self._normalise:
            reps = F.normalize(reps, dim=-1)
        return reps

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        """Encode fingerprints with optional support dimensions."""

        if inputs.dim() == 2:
            return self._encode_flat(inputs)
        if inputs.dim() == 3:
            bsz, support, feat = inputs.shape
            flat = inputs.view(bsz * support, feat)
            reps = self._encode_flat(flat)
            return reps.view(bsz, support, -1)
        raise ValueError(
            "TabICLEncoder expects tensors of shape (batch, features) "
            "or (batch, support, features)."
        )


__all__ = ["TabICLCheckpointConfig", "TabICLEncoder"]
