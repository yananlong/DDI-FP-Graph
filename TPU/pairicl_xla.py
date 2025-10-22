"""PyTorch/XLA PairICL models using pretrained TabICL encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tabicl_xla import TabICLCheckpointConfig, TabICLEncoder


def _activation_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation '{name}'.")


def _pairwise_feature_map(lhs: Tensor, rhs: Tensor) -> Tensor:
    return torch.cat([lhs, rhs, torch.abs(lhs - rhs), lhs * rhs], dim=-1)


@dataclass
class PairICLConfig:
    """Configuration for the PairICL projection head."""

    hidden_dim: int = 1024
    projection_dim: int = 512
    encoder_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.1
    temperature: float = 0.07
    support_blend: float = 0.5
    use_support: bool = False


class PairICLHead(nn.Module):
    """Prototype-based classifier supporting zero-shot and support modes."""

    def __init__(
        self,
        *,
        num_classes: int,
        projection_dim: int,
        temperature: float,
        support_blend: float,
        use_support: bool,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._support_blend = float(support_blend)
        self._use_support = bool(use_support)
        self._temperature = float(max(temperature, 1e-6))

        self.base_prompts = nn.Parameter(torch.empty(num_classes, projection_dim))
        nn.init.trunc_normal_(self.base_prompts, std=0.02)
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        pair_repr: Tensor,
        *,
        support_repr: Optional[Tensor] = None,
        support_labels: Optional[Tensor] = None,
        support_mask: Optional[Tensor] = None,
    ) -> Tensor:
        base_prompts = self.base_prompts
        if self._use_support and support_repr is not None and support_labels is not None:
            support_repr = F.normalize(support_repr, dim=-1, eps=1e-6)
            support_labels = support_labels.long()
            if support_mask is None:
                mask = torch.ones_like(support_labels, dtype=support_repr.dtype)
            else:
                mask = support_mask.to(support_repr.dtype)
            mask = mask.unsqueeze(-1)
            one_hot = F.one_hot(support_labels, num_classes=self._num_classes).to(support_repr.dtype)
            weighted = one_hot * mask
            support_counts = weighted.sum(dim=1)
            weights = support_counts.clamp_min(1.0).unsqueeze(-1)
            prototypes = torch.matmul(weighted.transpose(1, 2), support_repr) / weights
            has_support = (support_counts > 0).unsqueeze(-1)
            blend = self._support_blend * has_support.to(prototypes.dtype)
            prompts = base_prompts.unsqueeze(0) * (1.0 - blend) + prototypes * blend
        else:
            prompts = base_prompts.unsqueeze(0)

        pair_repr = F.normalize(pair_repr, dim=-1, eps=1e-6).unsqueeze(1)
        prompts = F.normalize(prompts, dim=-1, eps=1e-6)
        logits = torch.matmul(pair_repr, prompts.transpose(1, 2)).squeeze(1)
        scale = torch.exp(self.logit_scale) / self._temperature
        return logits * scale


class PairICLEncoder(nn.Module):
    """Shared feature extractor for queries and optional support sets."""

    def __init__(self, tabicl: TabICLEncoder, config: PairICLConfig) -> None:
        super().__init__()
        self.tabicl = tabicl
        self.config = config
        self.embedding_dim = tabicl.output_dim

        in_dim = self.embedding_dim * 4
        layers: list[nn.Module] = []
        hidden_dim = config.hidden_dim
        act = _activation_fn(config.activation)
        for _ in range(max(config.encoder_layers - 1, 0)):
            layers.append(nn.Linear(in_dim if not layers else hidden_dim, hidden_dim))
            layers.append(act)
            if config.dropout:
                layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(hidden_dim if layers else in_dim, config.projection_dim))
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        self.projector = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(config.projection_dim)

    def forward(
        self,
        fp1: Tensor,
        fp2: Tensor,
        *,
        support_fp1: Optional[Tensor] = None,
        support_fp2: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        batch = fp1.size(0)
        inputs = [fp1, fp2]
        splits = [batch, batch]

        support_repr: Optional[Tensor] = None
        support_shape: Optional[tuple[int, int, int]] = None
        if support_fp1 is not None and support_fp2 is not None:
            bsz, supp, feat = support_fp1.shape
            support_shape = (bsz, supp, feat)
            flat_support_fp1 = support_fp1.reshape(bsz * supp, feat)
            flat_support_fp2 = support_fp2.reshape(bsz * supp, feat)
            inputs.extend([flat_support_fp1, flat_support_fp2])
            splits.extend([flat_support_fp1.size(0), flat_support_fp2.size(0)])

        encoded = self.tabicl(torch.cat(inputs, dim=0))
        emb_chunks = list(encoded.split(splits, dim=0))
        emb1, emb2 = emb_chunks[0], emb_chunks[1]
        pair_repr = self.norm(self.projector(_pairwise_feature_map(emb1, emb2)))

        if support_shape is not None:
            bsz, supp, _ = support_shape
            supp_emb1 = emb_chunks[2].view(bsz, supp, -1)
            supp_emb2 = emb_chunks[3].view(bsz, supp, -1)
            support_repr = self.norm(
                self.projector(_pairwise_feature_map(supp_emb1, supp_emb2))
            )
        return pair_repr, support_repr


class PairICLModel(nn.Module):
    """PairICL classifier that optionally consumes episodic support sets."""

    def __init__(
        self,
        *,
        tabicl_config: TabICLCheckpointConfig,
        pairicl_config: PairICLConfig,
        num_classes: int,
    ) -> None:
        super().__init__()
        tabicl = TabICLEncoder(tabicl_config)
        self.encoder = PairICLEncoder(tabicl, pairicl_config)
        self.head = PairICLHead(
            num_classes=num_classes,
            projection_dim=pairicl_config.projection_dim,
            temperature=pairicl_config.temperature,
            support_blend=pairicl_config.support_blend,
            use_support=pairicl_config.use_support,
        )
        self.use_support = pairicl_config.use_support

    def forward(
        self,
        fp1: Tensor,
        fp2: Tensor,
        *,
        support_fp1: Optional[Tensor] = None,
        support_fp2: Optional[Tensor] = None,
        support_labels: Optional[Tensor] = None,
        support_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pair_repr, support_repr = self.encoder(
            fp1,
            fp2,
            support_fp1=support_fp1 if self.use_support else None,
            support_fp2=support_fp2 if self.use_support else None,
        )
        return self.head(
            pair_repr,
            support_repr=support_repr,
            support_labels=support_labels if self.use_support else None,
            support_mask=support_mask if self.use_support else None,
        )


def build_pairicl_zero_shot(
    *, tabicl_config: TabICLCheckpointConfig, pairicl_config: PairICLConfig, num_classes: int
) -> PairICLModel:
    cfg = PairICLConfig(**{**pairicl_config.__dict__, "use_support": False})
    return PairICLModel(tabicl_config=tabicl_config, pairicl_config=cfg, num_classes=num_classes)


def build_pairicl_support(
    *, tabicl_config: TabICLCheckpointConfig, pairicl_config: PairICLConfig, num_classes: int
) -> PairICLModel:
    cfg = PairICLConfig(**{**pairicl_config.__dict__, "use_support": True})
    return PairICLModel(tabicl_config=tabicl_config, pairicl_config=cfg, num_classes=num_classes)


__all__ = [
    "PairICLConfig",
    "PairICLHead",
    "PairICLEncoder",
    "PairICLModel",
    "build_pairicl_zero_shot",
    "build_pairicl_support",
]
