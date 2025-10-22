"""PyTorch/XLA training entry point for PairICL models."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from .data_xla import NPZPairDataset, PairICLEpisodeDataset, load_metadata, pairicl_collate
from .pairicl_xla import PairICLConfig, PairICLModel
from .tabicl_xla import TabICLCheckpointConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/val/test NPZ splits.")
    parser.add_argument("--tabicl-checkpoint", type=Path, required=True, help="Path to the pretrained TabICL checkpoint.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--support-size", type=int, default=0)
    parser.add_argument("--support-blend", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=["bf16", "f32"],
        default="bf16",
        help="Compute precision to use on TPU cores.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of batches prefetched by each dataloader worker (ignored when num_workers=0).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable in-memory caching of NPZ samples to reduce host RAM usage.",
    )
    parser.add_argument("--train-tabicl", action="store_true", help="Allow fine-tuning the TabICL encoder.")
    parser.add_argument("--no-normalise", action="store_true", help="Disable L2 normalisation on TabICL outputs.")
    parser.add_argument("--num-cores", type=int, default=8, help="Number of TPU cores to spawn.")
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_dataloader(
    root: Path,
    *,
    support_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    cache_dataset: bool,
    prefetch_factor: int,
    dtype: torch.dtype,
) -> tuple[DataLoader, DistributedSampler]:
    base_dataset = NPZPairDataset(root, cache=cache_dataset, dtype=dtype)
    episode_dataset = PairICLEpisodeDataset(base_dataset, support_size)
    sampler = DistributedSampler(
        episode_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle,
        seed=seed,
    )

    def _worker_init(worker_id: int) -> None:
        worker_seed = seed + xm.get_ordinal() * 1000 + worker_id
        _seed_everything(worker_seed)

    loader_kwargs: dict[str, object] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(
        episode_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=pairicl_collate,
        worker_init_fn=_worker_init,
        **loader_kwargs,
    )
    return loader, sampler


def _evaluate(
    model: PairICLModel,
    loader: pl.MpDeviceLoader,
    device: torch.device,
    *,
    precision_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    correct = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            with torch.autocast("xla", dtype=precision_dtype, enabled=use_amp):
                logits = model(
                    batch["fp1"],
                    batch["fp2"],
                    support_fp1=batch.get("support_fp1"),
                    support_fp2=batch.get("support_fp2"),
                    support_labels=batch.get("support_labels"),
                    support_mask=batch.get("support_mask"),
                )
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum()
            total += labels.size(0)
            xm.mark_step()
    correct = xm.all_reduce(xm.REDUCE_SUM, correct)
    total = xm.all_reduce(xm.REDUCE_SUM, total)
    xm.mark_step()
    return (correct / total).item()


def _train_worker(index: int, args: argparse.Namespace) -> None:  # pragma: no cover - executed on TPU
    del index
    device = xm.xla_device()
    _seed_everything(args.seed + xm.get_ordinal())

    precision_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    use_amp = precision_dtype == torch.bfloat16

    metadata = load_metadata(args.data_dir / "metadata.json")
    num_classes = int(metadata["num_classes"])

    tabicl_cfg = TabICLCheckpointConfig(
        checkpoint_path=args.tabicl_checkpoint,
        trainable=args.train_tabicl,
        normalise=not args.no_normalise,
        dtype=precision_dtype,
    )
    pairicl_cfg = PairICLConfig(
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        encoder_layers=args.encoder_layers,
        activation=args.activation,
        dropout=args.dropout,
        temperature=args.temperature,
        support_blend=args.support_blend,
        use_support=args.support_size > 0,
    )

    model = PairICLModel(tabicl_config=tabicl_cfg, pairicl_config=pairicl_cfg, num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    cache_dataset = not args.no_cache
    train_loader, train_sampler = _create_dataloader(
        args.data_dir / "train",
        support_size=args.support_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
        cache_dataset=cache_dataset,
        prefetch_factor=args.prefetch_factor,
        dtype=precision_dtype,
    )
    val_loader, _ = _create_dataloader(
        args.data_dir / "val",
        support_size=args.support_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
        cache_dataset=cache_dataset,
        prefetch_factor=args.prefetch_factor,
        dtype=precision_dtype,
    )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_device_loader):
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"]
            with torch.autocast("xla", dtype=precision_dtype, enabled=use_amp):
                logits = model(
                    batch["fp1"],
                    batch["fp2"],
                    support_fp1=batch.get("support_fp1"),
                    support_fp2=batch.get("support_fp2"),
                    support_labels=batch.get("support_labels"),
                    support_mask=batch.get("support_mask"),
                )
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            xm.mark_step()

            if step % args.log_steps == 0:
                loss_val = xm.all_reduce(xm.REDUCE_MEAN, loss.detach())
                if xm.is_master_ordinal():
                    print(f"epoch={epoch} step={step} loss={loss_val.item():.4f}")
        val_accuracy = _evaluate(
            model,
            val_device_loader,
            device,
            precision_dtype=precision_dtype,
            use_amp=use_amp,
        )
        if xm.is_master_ordinal():
            print(f"epoch={epoch} val_accuracy={val_accuracy:.4f}")

    if xm.is_master_ordinal():
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
            "config": {
                "pairicl": vars(pairicl_cfg),
                "tabicl": {
                    "checkpoint_path": str(args.tabicl_checkpoint),
                    "trainable": args.train_tabicl,
                    "normalise": not args.no_normalise,
                    "dtype": args.precision,
                },
                "precision": args.precision,
            },
        }
        torch.save(state, args.data_dir / "pairicl_xla.ckpt")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    xmp.spawn(_train_worker, args=(args,), nprocs=args.num_cores)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
