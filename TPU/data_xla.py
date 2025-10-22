"""Data utilities for the PyTorch/XLA PairICL pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


def load_metadata(path: Path) -> Dict[str, object]:
    """Load the metadata exported by ``preprocess_to_npz``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class NPZPairDataset(Dataset):
    """Dataset reading fingerprint pairs and labels from compressed NPZ files."""

    def __init__(self, root: Path, *, cache: bool = True, dtype: torch.dtype = torch.float32) -> None:
        self.root = Path(root)
        if not self.root.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"Dataset directory {self.root} does not exist")
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:  # pragma: no cover - defensive
            raise ValueError(f"No samples found in {self.root}")

        self.dtype = dtype
        self._cache_enabled = bool(cache)
        self._cached: List[Dict[str, torch.Tensor]] | None = [] if self._cache_enabled else None
        self.labels: List[int] = []
        for file in self.files:
            with np.load(file, allow_pickle=False) as data:
                label = int(data["label"])
                self.labels.append(label)
                if self._cache_enabled:
                    self._cached.append(self._convert_npz(data, label))

    def _convert_npz(self, arrays: Dict[str, np.ndarray], label: int) -> Dict[str, torch.Tensor]:
        fp1 = torch.from_numpy(arrays["fp1"]).to(self.dtype).view(-1)
        fp2 = torch.from_numpy(arrays["fp2"]).to(self.dtype).view(-1)
        return {"fp1": fp1, "fp2": fp2, "label": torch.tensor(label, dtype=torch.long)}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cached is not None:
            return dict(self._cached[idx])

        file = self.files[idx]
        with np.load(file, allow_pickle=False) as data:
            label = int(data["label"])
            return self._convert_npz(data, label)


class PairICLEpisodeDataset(Dataset):
    """Wrap ``NPZPairDataset`` to optionally attach support examples per query."""

    def __init__(self, base: NPZPairDataset, support_size: int) -> None:
        self.base = base
        self.support_size = max(0, int(support_size))
        by_class: Dict[int, List[int]] = {}
        for index, label in enumerate(base.labels):
            by_class.setdefault(label, []).append(index)
        self._class_indices: Dict[int, torch.Tensor] = {
            label: torch.tensor(indices, dtype=torch.int64)
            for label, indices in by_class.items()
        }

    def __len__(self) -> int:
        return len(self.base)

    def _sample_support_indices(self, label: int, exclude: int) -> torch.Tensor:
        if self.support_size == 0:
            return torch.empty(0, dtype=torch.int64)

        indices = self._class_indices[label]
        if indices.numel() == 0:
            return torch.empty(0, dtype=torch.int64)

        mask = indices != exclude
        candidates = indices[mask]
        if candidates.numel() == 0:
            candidates = indices

        if candidates.numel() >= self.support_size:
            perm = torch.randperm(candidates.numel())[: self.support_size]
            return candidates[perm]

        repeat_factor = (self.support_size + candidates.numel() - 1) // candidates.numel()
        expanded = candidates.repeat(repeat_factor)
        return expanded[: self.support_size]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = dict(self.base[idx])
        label = int(sample["label"])
        support_indices = self._sample_support_indices(label, idx)
        if support_indices.numel() > 0:
            support_fp1 = []
            support_fp2 = []
            support_labels = []
            for support_idx in support_indices.tolist():
                support_sample = self.base[support_idx]
                support_fp1.append(support_sample["fp1"])
                support_fp2.append(support_sample["fp2"])
                support_labels.append(int(support_sample["label"]))
            sample["support_fp1"] = torch.stack(support_fp1, dim=0)
            sample["support_fp2"] = torch.stack(support_fp2, dim=0)
            sample["support_labels"] = torch.tensor(support_labels, dtype=torch.long)
            sample["support_mask"] = torch.ones(len(support_labels), dtype=torch.bool)
        return sample


def pairicl_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of PairICL samples for PyTorch/XLA."""

    fp1 = torch.stack([item["fp1"] for item in batch], dim=0)
    fp2 = torch.stack([item["fp2"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    collated: Dict[str, torch.Tensor] = {"fp1": fp1, "fp2": fp2, "labels": labels}

    if "support_fp1" in batch[0] and batch[0]["support_fp1"] is not None:
        collated["support_fp1"] = torch.stack([item["support_fp1"] for item in batch], dim=0)
        collated["support_fp2"] = torch.stack([item["support_fp2"] for item in batch], dim=0)
        collated["support_labels"] = torch.stack([item["support_labels"] for item in batch], dim=0)
        collated["support_mask"] = torch.stack([item["support_mask"] for item in batch], dim=0)
    return collated


__all__ = [
    "load_metadata",
    "NPZPairDataset",
    "PairICLEpisodeDataset",
    "pairicl_collate",
]
