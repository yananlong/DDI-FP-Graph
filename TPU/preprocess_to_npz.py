"""Convert the PyTorch Geometric datasets to NumPy archives for TF-GNN training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from PyTorch.fp_data import FPGraphDataModule


def _iter_subset(subset) -> Iterable:
    if hasattr(subset, "indices"):
        dataset = subset.dataset
        indices = subset.indices
    else:
        dataset = subset
        indices = range(len(subset))
    for idx in indices:
        yield dataset[idx]


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def export_subset(name: str, subset, out_dir: Path) -> dict:
    subset_dir = out_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_examples": 0,
        "node_dim": None,
        "edge_dim": None,
        "fp_dim": None,
    }
    for idx, sample in enumerate(tqdm(_iter_subset(subset), desc=f"Exporting {name}")):
        data = {
            "label": int(sample.y),
            "fp1": _torch_to_numpy(sample.fp1),
            "fp2": _torch_to_numpy(sample.fp2),
            "x1": _torch_to_numpy(sample.x1),
            "x2": _torch_to_numpy(sample.x2),
            "edge_index1": _torch_to_numpy(sample.edge_index1),
            "edge_index2": _torch_to_numpy(sample.edge_index2),
            "edge_attr1": _torch_to_numpy(sample.edge_attr1),
            "edge_attr2": _torch_to_numpy(sample.edge_attr2),
        }
        if metadata["node_dim"] is None:
            metadata["node_dim"] = data["x1"].shape[1]
        if metadata["edge_dim"] is None:
            metadata["edge_dim"] = data["edge_attr1"].shape[1]
        if metadata["fp_dim"] is None:
            metadata["fp_dim"] = data["fp1"].shape[1]
        np.savez_compressed(subset_dir / f"{idx:06d}.npz", **data)
        metadata["num_examples"] += 1
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", type=str, default="morgan")
    parser.add_argument("--mode", type=str, default="transductive")
    parser.add_argument("--train-prop", type=float, default=0.8)
    parser.add_argument("--val-prop", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dm = FPGraphDataModule(
        root=str(args.data_dir),
        data_dir=str(args.data_dir),
        kind=args.kind,
        include_neg=True,
        mode=args.mode,
        train_prop=args.train_prop,
        val_prop=args.val_prop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    subsets = {"train": dm.train, "val": dm.val, "test": dm.test}
    metadata = {
        "num_classes": int(dm.num_classes),
        "splits": {},
        "mode": args.mode,
        "kind": args.kind,
    }
    for name, subset in subsets.items():
        subset_meta = export_subset(name, subset, args.output_dir)
        metadata["splits"][name] = subset_meta
        metadata.setdefault("node_dim", subset_meta.get("node_dim"))
        metadata.setdefault("edge_dim", subset_meta.get("edge_dim"))
        metadata.setdefault("fp_dim", subset_meta.get("fp_dim"))

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Export finished. Metadata saved to {(args.output_dir / 'metadata.json').resolve()}")


if __name__ == "__main__":
    main()
