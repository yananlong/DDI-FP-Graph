"""Command line entry point for training DDI-FP-Graph models with Lightning 2.x."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from lightning import pytorch as pl
import torch_geometric.nn as pyg_nn
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import (
    DataModuleConfig,
    ExperimentConfig,
    FPCatBoostConfig,
    FPXGBoostConfig,
    FPLightGBMConfig,
    FingerprintGraphModelConfig,
    FingerprintMLPConfig,
    GraphModelConfig,
    OptimizerConfig,
    SSIDDIModelConfig,
    TrainerConfig,
    WandbConfig,
)
from .fp_data import FPDataModule, FPGraphDataModule
from .models import (
    FPGraphModel,
    FPCatBoostModel,
    FPLightGBMModel,
    FPMLP,
    FPXGBoostModel,
    GraphModel,
    SSIDDIModel,
)


MODEL_REGISTRY = {
    "graph": GraphModel,
    "fp_mlp": FPMLP,
    "fp_catboost": FPCatBoostModel,
    "fp_lightgbm": FPLightGBMModel,
    "fp_xgboost": FPXGBoostModel,
    "fp_graph": FPGraphModel,
    "ssi_ddi": SSIDDIModel,
}
MODEL_REGISTRY["fp"] = MODEL_REGISTRY["fp_mlp"]

MODEL_CONFIGS = {
    "graph": GraphModelConfig,
    "fp_mlp": FingerprintMLPConfig,
    "fp_catboost": FPCatBoostConfig,
    "fp_lightgbm": FPLightGBMConfig,
    "fp_xgboost": FPXGBoostConfig,
    "fp_graph": FingerprintGraphModelConfig,
    "ssi_ddi": SSIDDIModelConfig,
}
MODEL_CONFIGS["fp"] = MODEL_CONFIGS["fp_mlp"]


def _resolve_gnn_layer(name: str) -> type[pyg_nn.MessagePassing]:
    try:
        return getattr(pyg_nn, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown torch_geometric layer: {name}") from exc


def _load_yaml(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise TypeError("Configuration file must decode to a mapping")
    return data


def _build_experiment_config(raw: Dict[str, Any]) -> ExperimentConfig:
    trainer_cfg = TrainerConfig(**raw.get("trainer", {}))
    wandb_cfg = WandbConfig(**raw.get("wandb", {}))
    optim_cfg = OptimizerConfig(**raw.get("optimizer", {}))
    datamodule_cfg = DataModuleConfig(**raw.get("datamodule", {}))
    model_cfg = raw.get("model", {})
    model_name = raw.get("model_name", "graph")
    seed = raw.get("seed", 2023)
    return ExperimentConfig(
        trainer=trainer_cfg,
        wandb=wandb_cfg,
        optimizer=optim_cfg,
        datamodule=datamodule_cfg,
        model=model_cfg,
        seed=seed,
        model_name=model_name,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--model",
        choices=MODEL_REGISTRY.keys(),
        default=None,
        help="Override the model specified in the config file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory used by the LightningDataModule.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Explicit name for the Weights & Biases run.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Set WANDB_MODE for logging (defaults to the environment value).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resume training from an existing checkpoint.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch of train/val/test for debugging.",
    )
    return parser.parse_args()


def _instantiate_datamodule(name: str, cfg: DataModuleConfig) -> FPDataModule | FPGraphDataModule:
    fingerprint_kwargs = dict(radius=cfg.radius, nBits=cfg.nbits)
    common_kwargs = dict(
        kind=cfg.kind,
        include_neg=cfg.include_neg,
        mode=cfg.mode,
        train_prop=cfg.train_prop,
        val_prop=cfg.val_prop,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        **fingerprint_kwargs,
    )
    if common_kwargs["num_workers"] and common_kwargs["num_workers"] > 0 and cfg.mode.startswith("inductive"):
        # Inductive splits benefit from deterministic sampling; keep worker count low.
        common_kwargs["num_workers"] = max(1, common_kwargs["num_workers"])
    if name in {"graph", "fp_graph", "ssi_ddi"}:
        return FPGraphDataModule(
            root=str(cfg.data_dir),
            data_dir=str(cfg.data_dir),
            **common_kwargs,
        )
    fingerprint_models = {"fp", "fp_mlp", "fp_catboost", "fp_lightgbm", "fp_xgboost"}
    if name in fingerprint_models:
        return FPDataModule(data_dir=str(cfg.data_dir), **common_kwargs)
    raise ValueError(f"Unsupported model name for datamodule instantiation: {name}")


def _build_model(
    name: str,
    cfg: ExperimentConfig,
    dm: FPDataModule | FPGraphDataModule,
) -> pl.LightningModule:
    normalized_name = "fp_mlp" if name == "fp" else name
    model_cfg_cls = MODEL_CONFIGS[normalized_name]
    model_cfg = model_cfg_cls(**cfg.model)
    optimizer_name = cfg.optimizer.name
    opt_kwargs = dict(lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, optimizer=optimizer_name)

    if normalized_name == "graph":
        gnn_layer = _resolve_gnn_layer(model_cfg.gnn_name)
        return GraphModel(
            batch_size=cfg.datamodule.batch_size,
            act=model_cfg.act,
            gnn_name=gnn_layer,
            gnn_nlayers=model_cfg.gnn_nlayers,
            gnn_in=model_cfg.gnn_in,
            gnn_hid=model_cfg.gnn_hid,
            dec_nlayers=model_cfg.dec_nlayers,
            dec_hid=model_cfg.dec_hid,
            attn_heads=model_cfg.attn_heads,
            out_dim=dm.num_classes,
            final_concat=model_cfg.final_concat,
            dropout=model_cfg.dropout,
            top_k=model_cfg.top_k,
            **opt_kwargs,
        )
    if normalized_name == "fp_mlp":
        return FPMLP(
            in_dim=dm.ndim,
            hid_dim=model_cfg.hid_dim,
            out_dim=dm.num_classes,
            nlayers=model_cfg.enc_layers,
            dropout=model_cfg.dropout,
            act=model_cfg.act,
            batch_norm=model_cfg.batch_norm,
            fusion=model_cfg.fusion,
            concat=model_cfg.concat,
            top_k=model_cfg.top_k,
            **opt_kwargs,
        )
    if normalized_name == "fp_catboost":
        return FPCatBoostModel(
            out_dim=dm.num_classes,
            fusion=model_cfg.fusion,
            top_k=model_cfg.top_k,
            depth=model_cfg.depth,
            learning_rate=model_cfg.learning_rate,
            iterations=model_cfg.iterations,
            l2_leaf_reg=model_cfg.l2_leaf_reg,
            bagging_temperature=model_cfg.bagging_temperature,
            random_strength=model_cfg.random_strength,
            random_state=model_cfg.random_state,
            estimator_kwargs=model_cfg.extra_params,
            device=model_cfg.device,
        )
    if normalized_name == "fp_lightgbm":
        return FPLightGBMModel(
            out_dim=dm.num_classes,
            fusion=model_cfg.fusion,
            top_k=model_cfg.top_k,
            num_leaves=model_cfg.num_leaves,
            learning_rate=model_cfg.learning_rate,
            n_estimators=model_cfg.n_estimators,
            subsample=model_cfg.subsample,
            colsample_bytree=model_cfg.colsample_bytree,
            min_child_samples=model_cfg.min_child_samples,
            reg_alpha=model_cfg.reg_alpha,
            reg_lambda=model_cfg.reg_lambda,
            random_state=model_cfg.random_state,
            estimator_kwargs=model_cfg.extra_params,
            device=model_cfg.device,
        )
    if normalized_name == "fp_xgboost":
        return FPXGBoostModel(
            out_dim=dm.num_classes,
            fusion=model_cfg.fusion,
            top_k=model_cfg.top_k,
            max_depth=model_cfg.max_depth,
            learning_rate=model_cfg.learning_rate,
            n_estimators=model_cfg.n_estimators,
            subsample=model_cfg.subsample,
            colsample_bytree=model_cfg.colsample_bytree,
            reg_lambda=model_cfg.reg_lambda,
            gamma=model_cfg.gamma,
            min_child_weight=model_cfg.min_child_weight,
            reg_alpha=model_cfg.reg_alpha,
            random_state=model_cfg.random_state,
            estimator_kwargs=model_cfg.extra_params,
            device=model_cfg.device,
        )
    if normalized_name == "fp_graph":
        gnn_layer = _resolve_gnn_layer(model_cfg.gnn_name)
        return FPGraphModel(
            batch_size=cfg.datamodule.batch_size,
            act=model_cfg.act,
            fp_nlayers=model_cfg.fp_nlayers,
            fp_in=model_cfg.fp_in or dm.ndim,
            fp_hid=model_cfg.fp_hid,
            gnn_name=gnn_layer,
            gnn_nlayers=model_cfg.gnn_nlayers,
            gnn_in=model_cfg.gnn_in,
            gnn_hid=model_cfg.gnn_hid,
            dec_nlayers=model_cfg.dec_nlayers,
            dec_hid=model_cfg.dec_hid,
            out_dim=dm.num_classes,
            final_concat=model_cfg.final_concat,
            dropout=model_cfg.dropout,
            top_k=model_cfg.top_k,
            **opt_kwargs,
        )
    if name == "ssi_ddi":
        return SSIDDIModel(
            batch_size=cfg.datamodule.batch_size,
            act=model_cfg.act,
            in_dim=model_cfg.in_dim,
            hid_dim=model_cfg.hid_dim,
            GAT_head_dim=model_cfg.GAT_head_dim,
            GAT_nheads=model_cfg.GAT_nheads,
            GAT_nlayers=model_cfg.GAT_nlayers,
            out_dim=dm.num_classes,
            top_k=model_cfg.top_k,
            **opt_kwargs,
        )
    raise ValueError(f"Unknown model name: {name}")


def _setup_trainer(cfg: ExperimentConfig, logger: WandbLogger) -> pl.Trainer:
    trainer_cfg = cfg.trainer
    devices = trainer_cfg.devices if trainer_cfg.devices is not None else "auto"
    trainer_kwargs = dict(
        accelerator=trainer_cfg.accelerator,
        devices=devices,
        max_epochs=trainer_cfg.max_epochs,
        min_epochs=trainer_cfg.min_epochs,
        precision=trainer_cfg.precision,
        deterministic=trainer_cfg.deterministic,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        enable_checkpointing=trainer_cfg.enable_checkpointing,
        fast_dev_run=trainer_cfg.fast_dev_run,
        logger=logger,
        enable_progress_bar=True,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(Path("checkpoints") / cfg.model_name),
            filename="{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=max(3, trainer_cfg.min_epochs),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return pl.Trainer(**trainer_kwargs, callbacks=callbacks)


def _config_to_nested_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return {
        "trainer": asdict(cfg.trainer),
        "optimizer": asdict(cfg.optimizer),
        "datamodule": asdict(cfg.datamodule),
        "model": cfg.model,
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "wandb": asdict(cfg.wandb),
    }


def _merge_config_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = json.loads(json.dumps(base))  # deep copy that preserves basic types
    allowed_roots = {"trainer", "optimizer", "datamodule", "model", "seed", "model_name"}

    def _set_path(target: Dict[str, Any], path: list[str], value: Any) -> None:
        node = target
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        if key in allowed_roots:
            if isinstance(value, dict):
                merged[key] = {**merged.get(key, {}), **value}
            else:
                merged[key] = value
        elif "." in key:
            root, *rest = key.split(".")
            if root not in allowed_roots:
                continue
            _set_path(merged.setdefault(root, {}), rest, value)
    return merged


def _apply_overrides(cfg: ExperimentConfig, merged: Dict[str, Any]) -> ExperimentConfig:
    cfg.trainer = TrainerConfig(**merged.get("trainer", {}))
    cfg.optimizer = OptimizerConfig(**merged.get("optimizer", {}))
    cfg.datamodule = DataModuleConfig(**merged.get("datamodule", {}))
    cfg.model = merged.get("model", cfg.model)
    cfg.seed = merged.get("seed", cfg.seed)
    cfg.model_name = merged.get("model_name", cfg.model_name)
    return cfg


def main() -> None:
    args = _parse_args()
    raw_cfg = _load_yaml(args.config)
    cfg = _build_experiment_config(raw_cfg)

    if args.model is not None:
        cfg.model_name = args.model
    if args.data_dir is not None:
        cfg.datamodule.data_dir = str(args.data_dir)
    if args.fast_dev_run:
        cfg.trainer.fast_dev_run = True
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    run_name = args.run_name
    base_config = _config_to_nested_dict(cfg)
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        log_model=cfg.wandb.log_model,
        save_dir=cfg.wandb.save_dir,
        name=run_name,
        config=base_config,
    )

    merged_config = _merge_config_dict(base_config, dict(wandb_logger.experiment.config))
    cfg = _apply_overrides(cfg, merged_config)

    pl.seed_everything(cfg.seed, workers=True)

    datamodule = _instantiate_datamodule(cfg.model_name, cfg.datamodule)
    datamodule.setup()

    model = _build_model(cfg.model_name, cfg, datamodule)

    wandb_logger.watch(model, log="all", log_freq=cfg.trainer.log_every_n_steps)

    trainer = _setup_trainer(cfg, wandb_logger)

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
